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

num_wave_tables = 32
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
replace = !,_;
process = ["freq":replace, "gain":replace, "gate":replace -> wavetable_synth];
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
T = int(SAMPLE_RATE * 0.1)

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

# Extract `WT Pos` parameter
wt_pos_params = {'_dawdreamer/WT Pos': params['_dawdreamer/WT Pos']}
# Extract the fixed wavetables
fixed_params = {k: v for k, v in params.items() if k != '_dawdreamer/WT Pos'}
print(f"Model parameter initialization time: {time.time() - start_time:.2f} seconds")

# Print the initial parameters
print("Initial parameters (WT Pos only):")
print_parameters(wt_pos_params)

### Step 2: Create a Target Sound
def generate_saw_wave(frequency, duration, sample_rate):
    t = jnp.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    saw_wave = 2 * (t * frequency - jnp.floor(t * frequency + 0.5))
    return saw_wave

# Parameters
duration = T / SAMPLE_RATE  # Duration in seconds
sample_rate = SAMPLE_RATE  # Sample rate

# Generate the saw wave
target_sound = jnp.stack([
    generate_saw_wave(pitch_to_hz(pitch), duration, sample_rate) for pitch in pitches
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

# Modified train step function
@jax.jit
def train_step(state, x, y, fixed_params):
    def loss_fn(params):
        # Combine the fixed parameters with the trainable parameters
        combined_params = {**fixed_params, **params}
        pred = batched_model.apply({'params': combined_params}, x, T)
        loss = spectrogram_loss(pred, y)
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Initialize optimizer
learning_rate = 1e-2
tx = optax.adam(learning_rate)

# Create initial training state for WT Pos only
state = train_state.TrainState.create(
    apply_fn=batched_model.apply,
    params=wt_pos_params,
    tx=tx
)

# Training loop
print("Starting training loop...")
num_steps = 200
pbar = tqdm(range(num_steps))
losses = []
wt_pos_values = []
for step in pbar:
    step_start_time = time.time()
    state, loss = train_step(state, input_tensor, target_sound, fixed_params)
    losses.append(loss)
    step_end_time = time.time()
    pbar.set_description(f"Step {step}, Loss: {loss:.4f}, Step Time: {step_end_time - step_start_time:.2f} seconds")

    if step % 10 == 0:  # Print parameters every 10 steps
        wt_pos_value = state.params['_dawdreamer/WT Pos']
        wt_pos_values.append(wt_pos_value)
        print(f"Step {step}, Loss: {loss:.4f}, WT Pos value: {wt_pos_value:.6f}", end='\r')

# Final print to move to the next line after loop ends
print()

# Plot the loss over time
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss over time")
plt.show()

# Plot the WT Pos values over time
plt.plot(wt_pos_values)
plt.xlabel("Step")
plt.ylabel("WT Pos value")
plt.title("WT Pos over time")
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
target_audio = generate_saw_wave(pitch_to_hz(pitches[0]), 1, SAMPLE_RATE)
target_audio_np = np.array(target_audio)  # Convert to NumPy array

# TEMP
print_parameters(state.params)
# TEMP
# Generate synthesized audio
synth_params = {**fixed_params, **state.params}
synth_input = pitch_to_tensor(pitches[0], 1, one_second_samples, one_second_samples)
synth_audio = batched_model.apply({'params': synth_params}, synth_input[None, ...], one_second_samples)[0, 0]
synth_audio_np = np.array(synth_audio)  # Convert to NumPy array

# Save audio files
sf.write('target_saw.wav', target_audio_np, SAMPLE_RATE)
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

print("Audio files saved as 'target_saw.wav' and 'synthesized_audio.wav'")
print("Spectrogram visualization with detailed frequency labels complete. Check the displayed plot.")
