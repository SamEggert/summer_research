from functools import partial
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
from dawdreamer.faust import FaustContext
from dawdreamer.faust.box import *
from tqdm import tqdm
import time

jax.config.update('jax_platform_name', 'cpu')

SAMPLE_RATE = 44100

# Directory containing the .wav files
print("Loading wavetable files...")
start_time = time.time()
directory = "./MB_Saw"
file_names = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.wav')]
file_names.sort()
print(f"Loaded {len(file_names)} wavetable files in {time.time() - start_time:.2f} seconds")

num_wave_tables = 16
num_files = len(file_names)
wave_names = [file_names[i] for i in range(0, num_files, num_files//num_wave_tables)]

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

wavetable_synth = multiwavetable(wtpos, ridx)*env1*gain
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

# Define the FX module (stereo panner)
dsp_content = """
import("stdfaust.lib");
pan = hslider("pan", 0.5, 0, 1, .01);
process = sp.panner(pan);
"""

print("Converting FX module to JAX...")
start_time = time.time()
with FaustContext():
    box = boxFromDSP(dsp_content, [])
    module_name = 'FaustFX'
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])
custom_globals = {}
exec(jax_code, custom_globals)  # security risk!
FaustFX = custom_globals[module_name]
print(f"FX module conversion time: {time.time() - start_time:.2f} seconds")

# Create an Instrument module that calls the voice batch, sums them, and passes it to the FX module.
class PolyInstrument(nn.Module):
    sample_rate: int
    soundfile_dirs: list[str]

    @nn.compact
    def __call__(self, x, T: int) -> jnp.array:
        polyvoices = PolyVoice(sample_rate=self.sample_rate, soundfile_dirs=self.soundfile_dirs)
        audio = polyvoices(x, T)
        audio = jnp.sum(audio, axis=0)
        fx = FaustFX(sample_rate=self.sample_rate)
        audio = fx(audio, T)
        return audio

SAMPLE_RATE = 44100
BatchedInstrument = nn.vmap(PolyInstrument, in_axes=(0, None), variable_axes={'params': 0}, split_rngs={'params': True})
batched_model = BatchedInstrument(SAMPLE_RATE, soundfile_dirs=[str(Path(os.getcwd())), str(Path(os.getcwd())/"MB_Saw")])
jit_batched_inference = jax.jit(partial(batched_model.apply, mutable='intermediates'), static_argnums=[2])

# Pass a batched tensor of freq/gain/gate and a batch of parameters to the batched Instrument.
T = int(SAMPLE_RATE*0.1)

def pitch_to_hz(pitch):
    return 440.0*(2.0**((pitch - 69)/12.0))

def pitch_to_tensor(pitch, gain, note_dur, total_dur):
    freq = pitch_to_hz(pitch)
    tensor = jnp.zeros((3, total_dur))
    tensor = tensor.at[:2, :].set(jnp.array([freq, gain]).reshape(2, 1))
    tensor = tensor.at[2, :note_dur].set(jnp.array([1]))
    tensor = jnp.expand_dims(tensor, axis=0)
    return tensor

print("Creating input tensors...")
start_time = time.time()
input_tensor = jnp.stack([
    pitch_to_tensor(60, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(62, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(64, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(65, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(67, 1, int(SAMPLE_RATE*1.), T),
], axis=0)
print(f"Created input tensors in {time.time() - start_time:.2f} seconds")

BATCH_SIZE = input_tensor.shape[0]

key = random.PRNGKey(42)
key, subkey = random.split(key)
print("Initializing model parameters...")
start_time = time.time()
params = batched_model.init({'params': subkey}, input_tensor, T)['params']
params['VmapFaustVoice_0']['_dawdreamer/WT Pos'] = jnp.linspace(-1, 1, num=BATCH_SIZE)
print(f"Model parameter initialization time: {time.time() - start_time:.2f} seconds")

### Step 2: Create a Target Sound
def generate_saw_wave(frequency, duration, sample_rate):
    t = jnp.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    saw_wave = 2 * (t * frequency - jnp.floor(t * frequency + 0.5))
    return saw_wave

# Parameters
frequency = 440  # Frequency of the saw wave
duration = T / SAMPLE_RATE  # Duration in seconds
sample_rate = SAMPLE_RATE  # Sample rate

# Generate the saw wave
target_sound = generate_saw_wave(frequency, duration, sample_rate)


### Step 3: Define a Loss Function
def loss_fn(params, x, y):
    pred, mod_vars = jit_batched_inference({'params': params}, x, T)
    loss = jnp.mean(jnp.abs(pred - y))
    return loss, pred

### Step 4: Optimize Parameters
learning_rate = 1e-3
tx = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=batched_model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        pred = batched_model.apply({'params': params}, x, T)
        loss = jnp.mean(jnp.abs(pred - y))
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
num_steps = 100
pbar = tqdm(range(num_steps))
losses = []

print("Starting training loop...")
for step in pbar:
    step_start_time = time.time()
    state, loss = train_step(state, input_tensor, target_sound)
    losses.append(loss)
    step_end_time = time.time()
    pbar.set_description(f"Step {step}, Loss: {loss:.4f}, Step Time: {step_end_time - step_start_time:.2f} seconds")

# Plot the loss over time
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss over time")
plt.show()

# Generate the final audio using the optimized parameters
# print("Generating final audio...")
# final_audio = batched_model.apply({'params': state.params}, input_tensor, T)

# # Plot the generated audio
# plt.plot(final_audio[0])
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")
# plt.title("Generated Audio")
# plt.show()
