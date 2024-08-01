from functools import partial
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy.signal as jss
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
from dawdreamer.faust import FaustContext
from dawdreamer.faust.box import *
from tqdm import tqdm
from evosax import CMA_ES
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

jax.config.update('jax_platform_name', 'cpu')

SAMPLE_RATE = 44100

# Directory containing the .wav files
print("Loading wavetable files...")
start_time = time.time()
directory = "./MB_Saw"
file_names = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.wav')]
file_names.sort()
print(f"Loaded {len(file_names)} wavetable files in {time.time() - start_time:.2f} seconds")

num_wave_tables = 4
num_files = len(file_names)
wave_names = [file_names[i] for i in range(0, num_files, num_files // num_wave_tables)]

# Initialize the string to be inserted
insert_string = "wavetables ="
for wave in wave_names:
    tmp = f"\n    wavetable(soundfile(\"param:{wave}[url:{{'{wave}.wav'}}]\",1)),"
    insert_string = insert_string + tmp
insert_string = insert_string[:-1] + ";"

# Dynamically construct the Faust code
print("Constructing Faust code...")
start_time = time.time()
dsp_content = f"""
import("stdfaust.lib");
// `sf` is a soundfile primitive
wavetable(sf, idx) = it.lagrangeN(N, f_idx, par(i, N + 1, table(i_idx - int(N / 2) + i)))
with {{
    N = 3; // lagrange order
    SFC = outputs(sf); // 1 for the length, 1 for the SR, and the remainder are the channels
    S = 0, 0 : sf : ba.selector(0, SFC); // table size
    table(j) = int(ma.modulo(j, S)) : sf(0) : !, !, si.bus(SFC-2);
    f_idx = ma.frac(idx) + int(N / 2);
    i_idx = int(idx);
}};
{insert_string}
NUM_TABLES = outputs(wavetables);
// ---------------multiwavetable-----------------------------
// All wavetables are placed evenly apart from each other.
//
// Where:
// * `wt_pos`: wavetable position from [0-1].
// * `ridx`: read index. floating point [0 - (S-1)]
multiwavetable(wt_pos, ridx) = si.vecOp((wavetables_output, coeff), *) :> _
with {{
    wavetables_output = ridx <: wavetables;
    coeff = par(i, NUM_TABLES, max(0, 1-abs(i-wt_pos*(NUM_TABLES-1))));
}};
S = 2048; // the length of the wavecycles, which we decided elsewhere
wavetable_synth = multiwavetable(wtpos, ridx) * env1 * gain
with {{
    freq = hslider("freq [style:knob]", 200 , 50  , 1000, .01 );
    gain = hslider("gain [style:knob]", .5  , 0   , 1   , .01 );
    gate = button("gate");
    wtpos = hslider("WT Pos [style:knob]", 0   , 0    , 1   ,  .01);
    ridx = os.hsp_phasor(S, freq, ba.impulsify(gate), 0);
    env1 = en.adsr(.01, 0, 1, .1, gate);
}};

// 3-band equalizer
equalizer = fi.low_shelf(LL, FL) : fi.peak_eq(LM, FM, BM) : fi.high_shelf(LH, FH)
with {{
    LL = vslider("Low Shelf Gain [unit:dB]", 0, -12, 12, 0.1);
    FL = hslider("Low Shelf Freq [unit:Hz]", 250, 20, 1000, 1);

    LM = vslider("Mid Peak Gain [unit:dB]", 0, -12, 12, 0.1);
    FM = hslider("Mid Peak Freq [unit:Hz]", 1000, 100, 5000, 1);
    BM = hslider("Mid Peak Q", 1, 0.1, 10, 0.1);

    LH = vslider("High Shelf Gain [unit:dB]", 0, -12, 12, 0.1);
    FH = hslider("High Shelf Freq [unit:Hz]", 4000, 1000, 20000, 1);
}};

replace = !,_;
process = ["freq":replace, "gain":replace, "gate":replace -> wavetable_synth : equalizer];
"""
print(f"Constructed Faust code in {time.time() - start_time:.2f} seconds")


# Convert Faust code to JAX
print("Converting Faust code to JAX...")
start_time = time.time()
with FaustContext():
    DSP_DIR1 = str(Path(os.getcwd()))
    argv = ['-I', DSP_DIR1]
    box = boxFromDSP(dsp_content, argv)
    module_name = 'FaustVoice'
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])
custom_globals = {}
exec(jax_code, custom_globals)  # security risk!
MonoVoice = custom_globals[module_name]
print(f"Faust to JAX conversion time: {time.time() - start_time:.2f} seconds")

# Use jax.linen.vmap to batch several voices, sharing parameters and PRNG among them.
PolyVoice = nn.vmap(MonoVoice, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})

SAMPLE_RATE = 44100
soundfile_dirs = [
    str(Path(os.getcwd())),
    str(Path(os.getcwd()) / "MB_Saw"),
    '/mnt/c/Users/braun/GitHub/sam-eggert-summer_research/code/MB_Saw'
]
batched_model = PolyVoice(SAMPLE_RATE, soundfile_dirs=soundfile_dirs)
jit_batched_inference = jax.jit(partial(batched_model.apply, mutable='intermediates'), static_argnums=[2])

# Pass a batched tensor of freq/gain/gate and a batch of parameters to the batched Instrument.
T = int(SAMPLE_RATE * 1)

def pitch_to_hz(pitch):
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

def pitch_to_tensor(pitch, gain, note_dur, total_dur):
    freq = pitch_to_hz(pitch)
    tensor = jnp.zeros((3, total_dur))
    tensor = tensor.at[:2, :].set(jnp.array([freq, gain]).reshape(2, 1))
    tensor = tensor.at[2, :note_dur].set(jnp.array([1]))
    return tensor

# pitches = [60, 64, 65, 67, 69]
pitches = [69]

print("Creating input tensors...")
start_time = time.time()
input_tensor = jnp.stack([
    pitch_to_tensor(pitch, 1, T, T) for pitch in pitches
], axis=0)
print(f"Created input tensors in {time.time() - start_time:.2f} seconds")

BATCH_SIZE = input_tensor.shape[0]




def print_parameters(params, prefix="", level=0, max_level=2):
    if isinstance(params, dict):
        for key, value in params.items():
            print_parameters(value, prefix=f"{prefix}{key}/", level=level + 1, max_level=max_level)
    else:
        print(f"{prefix}: shape={params.shape}")
        if params.shape == ():  # scalar value
            print(f"{prefix} value: {params}")
        elif params.ndim == 1 or params.size <= 10:  # small arrays
            print(f"{prefix} values: {params}")
        else:
            # For larger arrays, plot the first few values
            plt.figure(figsize=(10, 1))
            plt.plot(params.flatten()[:min(params.size, 100)], 'o-')
            plt.title(f"{prefix} (first 100 values)")
            plt.show()



# Initialize model parameters
key = random.PRNGKey(42)
key, subkey = random.split(key)
print("Initializing model parameters...")
start_time = time.time()
variables = batched_model.init({'params': subkey}, input_tensor, T)
params = variables['params']

# Fix the shape of WT Pos parameter if it's a scalar
if params['_dawdreamer/WT Pos'].ndim == 0:
    params['_dawdreamer/WT Pos'] = jnp.expand_dims(params['_dawdreamer/WT Pos'], axis=(0, 1))

trainable_params = {
    '_dawdreamer/WT Pos': jnp.array(0.5),  # Midpoint between wavetables
    '_dawdreamer/Low Shelf Gain': jnp.array(-5.0),
    '_dawdreamer/Low Shelf Freq': jnp.array(250.0),  # 250 Hz
    '_dawdreamer/Mid Peak Gain': jnp.array(3.0),
    '_dawdreamer/Mid Peak Freq': jnp.array(1000.0),  # 1000 Hz
    '_dawdreamer/Mid Peak Q': jnp.array(1.0),
    '_dawdreamer/High Shelf Gain': jnp.array(-2.0),
    '_dawdreamer/High Shelf Freq': jnp.array(4000.0)  # 4000 Hz
}
# Extract the fixed wavetables
fixed_params = {k: v for k, v in params.items() if k != '_dawdreamer/WT Pos'}
print(f"Model parameter initialization time: {time.time() - start_time:.2f} seconds")

# Print the initial parameters
print("Initial parameters:")
print_parameters(trainable_params)







### Step 2: Create a Target Sound
def generate_synth_sound(frequency, duration, sample_rate):
    # Adjustable parameters - modify these directly in the function
    adjustable_params = {
        '_dawdreamer/WT Pos': jnp.array(0.5),
        '_dawdreamer/Low Shelf Gain': jnp.array(0),
        '_dawdreamer/Low Shelf Freq': jnp.array(0),
        '_dawdreamer/Mid Peak Gain': jnp.array(0),
        '_dawdreamer/Mid Peak Freq': jnp.array(0),
        '_dawdreamer/Mid Peak Q': jnp.array(0),
        '_dawdreamer/High Shelf Gain': jnp.array(0),
        '_dawdreamer/High Shelf Freq': jnp.array(0)
    }

    # Combine fixed_params with adjustable_params
    synth_params = {**fixed_params, **adjustable_params}

    num_samples = int(duration * sample_rate)
    synth_input = pitch_to_tensor(frequency, 1, num_samples, num_samples)

    synth_audio = batched_model.apply({'params': synth_params}, synth_input[None, ...], num_samples)

    return synth_audio[0, 0]


# Parameters
duration = T / SAMPLE_RATE  # Duration in seconds
sample_rate = SAMPLE_RATE  # Sample rate

# Generate the saw wave
target_sound = jnp.stack([
    generate_synth_sound(pitch_to_hz(pitch), duration, sample_rate) for pitch in pitches
], axis=0)

target_sound = jnp.expand_dims(target_sound, 1)
print('target sound shape: ', target_sound.shape)

def spectrogram(x, fft_size=2048, hop_length=512):
    """Compute the spectrogram of a signal using JAX and scipy.signal."""
    # Ensure the input is 2D (add channel dimension if needed)
    if x.ndim == 1:
        x = x[jnp.newaxis, :]

    # Compute STFT
    f, t, Zxx = jss.stft(x, fs=SAMPLE_RATE, nperseg=fft_size, noverlap=fft_size-hop_length, padded=False, boundary=None)

    # Compute magnitude spectrogram
    return jnp.abs(Zxx)

def spectrogram_loss(pred, target):
    """Compute the loss between two spectrograms."""
    pred_spec = spectrogram(pred)
    target_spec = spectrogram(target)
    return jnp.mean(jnp.abs(pred_spec - target_spec))






# Define the fitness function
def fitness_function(params, input_tensor, target_sound, fixed_params):
    # Combine fixed and trainable parameters
    combined_params = {**fixed_params, **dict(zip(trainable_params.keys(), params))}
    # Predict the output sound
    pred = batched_model.apply({'params': combined_params}, input_tensor, T)
    # Calculate the spectrogram loss
    loss = spectrogram_loss(pred, target_sound)
    return loss



# Set up the evolutionary strategy
num_dims = len(trainable_params)
popsize = 20
strategy = CMA_ES(popsize=popsize, num_dims=num_dims)
es_params = strategy.default_params
es_params = es_params.replace(clip_min=-1, clip_max=1)

# Initialize the strategy
rng = jax.random.PRNGKey(0)
state = strategy.initialize(rng)

# Run the optimization
num_generations = 50
losses = []
param_values = {k: [] for k in trainable_params.keys()}

print("Starting evolutionary optimization...")
for gen in range(num_generations):
    rng, rng_ask, rng_eval = jax.random.split(rng, 3)

    # Ask for new solutions
    x, state = strategy.ask(rng_ask, state, es_params)

    # Evaluate fitness
    fitness = jax.vmap(lambda params: fitness_function(params, input_tensor, target_sound, fixed_params))(x)

    # Tell the results back to the strategy
    state = strategy.tell(x, fitness, state, es_params)

    # Logging
    best_fitness = jnp.min(fitness)
    best_params = x[jnp.argmin(fitness)]
    losses.append(best_fitness)
    for i, key in enumerate(trainable_params.keys()):
        param_values[key].append(best_params[i])

    if gen % 10 == 0:
        print(f"Generation {gen}: Best fitness = {best_fitness}")
        # for key, value in zip(trainable_params.keys(), best_params):
        #     print(f"{key} = {value}")

# Final print to move to the next line after loop ends
print()


# Plot the loss over generations
plt.figure(figsize=(12, 6))
plt.plot(losses)
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.title("Loss over Generations")
plt.show()

# Plot the parameter values over generations
fig, axs = plt.subplots(len(trainable_params), 1, figsize=(12, 4*len(trainable_params)))
for i, (key, values) in enumerate(param_values.items()):
    axs[i].plot(values)
    axs[i].set_xlabel("Generation")
    axs[i].set_ylabel(key)
    axs[i].set_title(f"{key} over Generations")
plt.tight_layout()
plt.show()



###### Generation and Visualization ####################################

import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Generate 1 second of audio
one_second_samples = SAMPLE_RATE

# Generate target saw wave
target_audio = generate_synth_sound(pitch_to_hz(pitches[0]), 1, SAMPLE_RATE)
target_audio_np = np.array(target_audio)  # Convert to NumPy array

# Get the best parameters from the evolutionary optimization
best_params = {k: param_values[k][-1] for k in trainable_params.keys()}

# Print the best parameters
print("Best parameters:")
print_parameters(best_params)

# Generate synthesized audio
synth_params = {**fixed_params, **best_params}
synth_input = pitch_to_tensor(pitches[0], 1, one_second_samples, one_second_samples)
synth_audio = batched_model.apply({'params': synth_params}, synth_input[None, ...], one_second_samples)[0, 0]
synth_audio_np = np.array(synth_audio)  # Convert to NumPy array

# Save audio files
sf.write('target_audio.wav', target_audio_np, SAMPLE_RATE)
sf.write('synthesized_audio.wav', synth_audio_np, SAMPLE_RATE)


# Visualize spectrograms using librosa with more options
plt.figure(figsize=(14, 6))


# Function to plot spectrogram with customized options
def plot_spectrogram(audio, title, position):
    plt.subplot(1, 2, position)
    D = librosa.stft(audio, n_fft=4096, hop_length=1024)  # Increased n_fft for better frequency resolution
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db,
                                   sr=SAMPLE_RATE,
                                   x_axis='time',
                                   y_axis='hz',
                                   cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.ylim(20, SAMPLE_RATE / 2)  # Set y-axis limits from 20 Hz to Nyquist frequency

    # Set custom y-axis ticks and labels
    yticks = [20, 50, 100, 200, 440, 1000, 2000, 5000, 10000, 20000]
    plt.yticks(yticks, [str(y) for y in yticks])

    # Add a horizontal line at 440 Hz
    plt.axhline(y=440, color='r', linestyle='--', alpha=0.5)


# Plot target audio spectrogram
plot_spectrogram(target_audio_np, 'Target Spectrogram', 1)

# Plot synthesized audio spectrogram
plot_spectrogram(synth_audio_np, 'Synthesized Spectrogram', 2)

plt.tight_layout()
plt.show()

print("Audio files saved as 'target_audio.wav' and 'synthesized_audio.wav'")
print("Spectrogram visualization with detailed frequency labels complete. Check the displayed plot.")