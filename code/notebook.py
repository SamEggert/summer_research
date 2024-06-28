from functools import partial
import itertools
from pathlib import Path
import os

import jax
default_device = 'cpu'
jax.config.update('jax_platform_name', default_device)
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.training import train_state  # Useful dataclass to keep train state
from flax.core.frozen_dict import unfreeze
import optax

from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
from dawdreamer.faust.box import *

from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation

########################################################################

SAMPLE_RATE = 44100

########################################################################

faust_code = """
import("stdfaust.lib");
cutoff = hslider("cutoff", 440., 20., 20000., .01);
process = fi.lowpass(1, cutoff);
"""

########################################################################

with FaustContext():
    box = boxFromDSP(faust_code)

    assert box.inputs == 1
    assert box.outputs == 1

    # Now we can convert it to C++ and specify a C++ class name MyDSP:
    module_name = "MyDSP"
    cpp_code = boxToSource(box, 'cpp', module_name)

# Or alternatively
createLibContext()
box = boxFromDSP(faust_code)
cpp_code = boxToSource(box, 'cpp', "MyDSP")
# print(cpp_code)
destroyLibContext()

########################################################################

with FaustContext():
    box = boxFromDSP(faust_code)

    assert box.inputs == 1
    assert box.outputs == 1

    module_name = "MyDSP"
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])
    # print(jax_code)

########################################################################

def faust2jax(faust_code: str):
    """
    Convert faust code into a batched JAX model and a single-item inference function.

    Inputs:
    * faust_code: string of faust code.
    """

    module_name = "MyDSP"
    with FaustContext():

      box = boxFromDSP(faust_code)

      jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

    custom_globals = {}

    exec(jax_code, custom_globals)  # security risk!

    MyDSP = custom_globals[module_name]
    # MyDSP is now a class definition which can be instantiated or subclassed.

    """
    In our vmap code, we specify `in_axes=(0, None)` to batch along `x` but not along `T`.
    In other words, the `x` argument will be a batch, but our `T` will remain a simple integer (not a batch of integers)
    We choose `variable_axes={'params': None, 'intermediates': 0}` in order to share parameters among the batch and continue to sow intermediate vars
    We choose `split_rngs={'params': False}` to use the same random number generator for each item in the batch.
    """
    MyDSP = nn.vmap(MyDSP, in_axes=(0, None), variable_axes={'params': None, 'intermediates': 0}, split_rngs={'params': False})

    # Now we can create a model that handles batches of input.
    model_batch = MyDSP(sample_rate=SAMPLE_RATE)

    # let's jit compile the model's `apply` method
    jit_inference_fn = jax.jit(partial(model_batch.apply, mutable='intermediates'), static_argnums=[2])

    # We jitted the model's "apply" function, which is of the form `apply(params, x, T)`.
    # T (the number of samples of the output) is a constant, so we specified static_argnums=[2].

    # We specify mutable='intermediates' to access intermediate variables, which are
    # human-interpretable.
    # Our architecture file normalized all of the parameters to be between -1 and 1.
    # During a forward pass, the parameters are remapped to their original ranges
    # and stored as intermediate variables via the `sow` method.

    return model_batch, jit_inference_fn

hidden_model, jit_hidden = faust2jax(faust_code)

########################################################################

# T is the number of audio samples of input and output
T = int(SAMPLE_RATE*1.0)  # 1 second of audio

batch_size = 8

# The middle dimension is the number of channels of input
input_shape = (batch_size, hidden_model.getNumInputs(), T)

key = random.PRNGKey(42)

key, subkey = random.split(key)
noises = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

key, subkey = random.split(key)
params = hidden_model.init({'params': subkey}, noises, T)

# print('normalized params: ', params)

audio, mod_vars = jit_hidden(params, noises, T)
# print('un-normalized params: ', mod_vars['intermediates'])

########################################################################

# Repeat the code much earlier, except create a model whose cutoff is 10,000 Hz.
init_cutoff = 10000 #@param {type: 'number'}
faust_code = f"""
import("stdfaust.lib");
cutoff = hslider("cutoff", {init_cutoff}, 20., 20000., .01);
process = fi.lowpass(1, cutoff);
"""

train_model, jit_train_inference = faust2jax(faust_code)

batch_size = 2 #@param {type: 'integer'}
input_shape = (batch_size, train_model.getNumInputs(), T)

# Create some noises that will serve as our validation dataset
key, subkey = random.split(key)
noises = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

key, subkey = random.split(key)
hidden_params = hidden_model.init({'params': subkey}, jnp.zeros_like(noises), T)['params']
key, subkey = random.split(key)
train_params = train_model.init({'params': subkey}, jnp.zeros_like(noises), T)['params']

# print('hidden params:', hidden_params)
# print('train params:', train_params)

learning_rate = 2e-4 #@param {type: 'number'}
momentum = 0.9 #@param {type: 'number'}

# Create Train state
tx = optax.sgd(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=train_model.apply, params=train_params, tx=tx)

@jax.jit
def train_step(state, x, y):
    """Train for a single step."""

    def loss_fn(params):
        pred = train_model.apply({'params': params}, x, T)
        # L1 time-domain loss
        loss = (jnp.abs(pred - y)).mean()
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

losses = []
cutoffs = []

train_steps = 3000
train_steps_per_eval = 100
pbar = tqdm(range(train_steps))

for n in pbar:
    # Generate a batch of inputs using our hidden parameters (440 Hz cutoff)
    key, subkey = random.split(key)
    x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)
    y, _ = jit_hidden({'params': hidden_params}, x, T)

    state, loss = train_step(state, x, y)

    if n % train_steps_per_eval == 0:
        # jit_train_inference accepts a batch of one item.
        audio, mod_vars = jit_train_inference({'params': state.params}, noises, T)
        cutoff = np.array(mod_vars['intermediates']['dawdreamer/cutoff'])
        # the cutoff above is a batch of predicted cutoff values, so we'll take the mean
        cutoff = cutoff.mean()
        losses.append(loss)
        cutoffs.append(cutoff)
        pbar.set_description(f"cutoff: {cutoff}")

# print('Done!')

########################################################################

plt.figure(figsize =(8, 4))

ax1 = plt.subplot(211)
ax1.plot(losses)
ax1.set_title("Evaluation Loss (L1 Time Domain)")
ax1.set_ylabel("Loss (Linear scale)")
plt.tick_params('x', labelbottom=False)

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(losses)
ax2.set_ylabel("Loss (Log scale)")
ax2.set_xlabel("Evaluation steps")
ax2.set_yscale('log')

plt.show()

plt.figure(figsize =(8, 4))

ax1 = plt.subplot(211)
ax1.plot(cutoffs)
ax1.set_title("Cutoff Parameter")
ax1.set_ylabel("Hz (Linear scale)")
plt.tick_params('x', labelbottom=False)

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(cutoffs)
ax2.set_ylabel("Hz (Log scale)")
ax2.set_xlabel("Evaluation steps")
ax2.set_yscale('log')

plt.show()

########################################################################

# Pick a single example
x = noises[0:1]
y = hidden_model.apply({'params': hidden_params}, x, T)

# Pick the first param to be the varying parameter.
# There happens to only be one parameter.
param_name = list(hidden_params.keys())[0]

@jax.jit
def loss_one_sample(params):
    pred, mod_vars = jit_train_inference({'params': params}, x, T)
    assert pred.shape == y.shape
    loss = jnp.abs(pred-y).mean()

    return loss, mod_vars['intermediates']

loss_many_samples = jax.vmap(loss_one_sample, in_axes=0, out_axes=0)

loss_landscape_batch = 500

batched_hidden_params = jax.tree_map(lambda x: jnp.tile(x, loss_landscape_batch), hidden_params)
batched_hidden_params = unfreeze(batched_hidden_params)
batched_hidden_params[param_name] = jnp.linspace(-1, 1, loss_landscape_batch)
landscape_losses, mod_vars = loss_many_samples(batched_hidden_params)

plt.figure(figsize=(6, 8))

ax1 = plt.subplot(2, 1, 1)
ax1.set_title("Loss Landscape")
ax1.plot(np.array(batched_hidden_params[param_name]), landscape_losses)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Normalized parameter')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(list(mod_vars.values())[0][0], landscape_losses)
ax2.set_ylabel('Loss')
ax2.set_xlabel(list(mod_vars.keys())[0])
ax2.scatter(cutoffs, losses, s=10, color='red', label="training")
ax2.legend()
plt.show()

########################################################################

# Pick our hidden parameters
benchmark_batch_size = 100  #@param {type:"integer"}
Duration_seconds = .1  #@param {type:"number"}

key, subkey = random.split(key)
noises = random.uniform(subkey, shape=(benchmark_batch_size, 1, int(SAMPLE_RATE*Duration_seconds)), minval=-1, maxval=1)

benchmark_jit = jax.jit(partial(train_model.apply), static_argnums=[2])

benchmark_noise = jnp.array(noises)

audio = benchmark_jit({'params': train_params}, benchmark_noise, T)

# print('benchmarking JAX:')
# %timeit benchmark_jit({'params': train_params}, benchmark_noise, T)

from scipy.signal import lfilter
import scipy.signal
b, a = scipy.signal.butter(1, 440, fs=SAMPLE_RATE)
noises_np = np.array(benchmark_noise)
# print('benchmarking scipy:')
noises_np = np.array(noises)
lfilter(b, a, noises_np)
# %timeit lfilter(b, a, noises_np)

########################################################################

def make_sine(freq: float, T: int, sr=SAMPLE_RATE):
    """Return sine wave based on freq in Hz and duration T in samples"""
    return jnp.sin(jnp.pi*2.*freq*jnp.arange(T)/sr)


faust_code = f"""
import("stdfaust.lib");
cutoff = hslider("cutoff", 440., 20., 20000., .01);
FX = fi.lowpass(1, cutoff);
replace = !,_;
process = ["cutoff":replace -> FX];
"""

module_name = "MyDSP"

with FaustContext():

  box = boxFromDSP(faust_code)

  jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

custom_globals = {}
exec(jax_code, custom_globals)
FilterModel = custom_globals[module_name]

# We don't vmap FilterModel in-place because we need the original version inside AutomationModel
HiddenModel = nn.vmap(FilterModel, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})

hidden_model = HiddenModel(SAMPLE_RATE)

class AutomationModel(nn.Module):

    automation_samples: int

    def getNumInputs(self):
        return 1

    @nn.compact
    def __call__(self, x, T: int) -> jnp.array:

        # make the learnable cutoff automation parameter.
        # It will start out as zero. We'll clamp it to [-1,1], and then remap to a useful range in Hz.
        cutoff = self.param('cutoff', nn.initializers.constant(0), (self.automation_samples,))
        # clamp the min to a safe range
        cutoff = jnp.clip(cutoff, -1., 1.)

        # Remap to a range in Hz that our DSP expects
        cutoff_min = 20
        cutoff_max = 20_000
        cutoff = jnp.interp(cutoff, jnp.array([-1., 1.]), jnp.array([cutoff_min, cutoff_max]))

        # Interpolate the cutoff to match the length of the input audio.
        # This is still differentiable.
        cutoff = jnp.interp(jnp.linspace(0,1,T), jnp.linspace(0,1,self.automation_samples), cutoff)

        # Sow our up-sampled cutoff automation, which we will access later when plotting.
        self.sow('intermediates', "cutoff", cutoff)

        # Expand dim to include channel
        cutoff = jnp.expand_dims(cutoff, axis=0)

        # Concatenate cutoff and input audio on the channel axis
        x = jnp.concatenate([cutoff, x], axis=-2)

        filterModel = FilterModel(sample_rate=SAMPLE_RATE)

        audio = filterModel(x, T)

        return audio

# set control rate to 100th of audio rate
AUTOMATION_DOWNSAMPLE = 100 #@param {type:"integer"}
RECORD_DURATION = 1.0 #@param {type:"number"}
T = int(RECORD_DURATION*SAMPLE_RATE)
automation_samples = T//AUTOMATION_DOWNSAMPLE

jit_inference_fn = jax.jit(partial(AutomationModel(automation_samples=automation_samples).apply, mutable='intermediates'), static_argnums=[2])
AutomationModel = nn.vmap(AutomationModel, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})

train_model = AutomationModel(automation_samples=automation_samples)

batch_size = 2 #@param {type: 'integer'}
hidden_automation_freq = 6.0 #@param {type:"number"}
hidden_automation = 10_000+make_sine(hidden_automation_freq, T)*9_500
jnp.expand_dims(hidden_automation, axis=0)
hidden_automation = jnp.tile(hidden_automation, (batch_size, 1, 1))
# print('hidden_automation shape: ', hidden_automation.shape)

########################################################################

input_shape = (batch_size, train_model.getNumInputs(), T)

key, subkey = random.split(key)
hidden_inputs = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

hidden_inputs = jnp.concatenate([hidden_automation, hidden_inputs], axis=1)

key, subkey = random.split(key)
hidden_params = hidden_model.init({'params': subkey}, hidden_inputs, T)

key, subkey = random.split(key)
x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

train_inputs = jnp.concatenate([jnp.zeros_like(hidden_automation), x], axis=1)
# print('train_inputs shape:', train_inputs.shape)

key, subkey = random.split(key)
train_params = train_model.init({'params': subkey}, train_inputs, T)['params']
# print('train params:', train_params)
# print('cutoff param shape:', train_params['cutoff'].shape)

########################################################################

learning_rate = 2e-1 #@param {type: 'number'}
momentum = 0.95 #@param {type: 'number'}

# Create Train state
tx = optax.sgd(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=train_model.apply, params=train_params, tx=tx)

@jax.jit
def train_step(state, x, y):
    """Train for a single step."""

    def loss_fn(params):
        pred = train_model.apply({'params': params}, x, T)
        # L1 time-domain loss
        loss = (jnp.abs(pred - y)).mean()
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

losses = []
cutoffs = []

train_steps = 600
steps_per_eval = 40

pbar = tqdm(range(train_steps))

jit_hidden_model = jax.jit(hidden_model.apply, static_argnums=[2])

# Data containers for animation
cutoff_data = []

for n in pbar:
    key, subkey = random.split(key)
    # generate random signal to be filtered
    x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

    # concat hidden automation to random signal
    automation_and_x = jnp.concatenate([hidden_automation, x], axis=1)

    # get ground truth of filtered signal
    y = jit_hidden_model(hidden_params, automation_and_x, T)

    # optimize learned automation based on input signal and GT filtered signal
    state, loss = train_step(state, x, y)

    if n % steps_per_eval == 0:
        audio, mod_vars = jit_inference_fn({'params': state.params}, x[0], T)
        cutoff = np.array(mod_vars['intermediates']['cutoff'])[0]
        cutoff_data.append(cutoff)
        pbar.set_description(f"loss: {loss:.2f}")
        losses.append(loss)

########################################################################

# Initialize the plot for animation
fig, ax = plt.subplots(figsize=(8, 4))
line1, = ax.plot([], [], label="Ground Truth")
line2, = ax.plot([], [], label="Prediction")
ax.set_title("Optimizing a Lowpass Filter's Cutoff")
ax.set_ylabel("Cutoff Frequency (Hz)")
ax.set_xlabel("Time (sec)")
ax.set_ylim(0,SAMPLE_RATE/2)
ax.set_xlim(0,T/SAMPLE_RATE)
plt.legend(loc='right')
time_axis = np.arange(T) / SAMPLE_RATE

# Function to update the plot for each frame
def update_plot(frame):
    global hidden_automation
    global cutoff_data
    line1.set_data(time_axis, hidden_automation[0, 0, :])
    line2.set_data(time_axis, cutoff_data[frame])
    return line1, line2

# Creating the animation
from matplotlib import rc
rc('animation', html='jshtml')
anim = animation.FuncAnimation(fig, update_plot, frames=len(cutoff_data), blit=True)
plt.close()

########################################################################

faust_code = """
import("stdfaust.lib");

eps = .00001;

synth = hgroup("Synth", result)
with {
    // UI groups
    VOICE(x) = hgroup("[0] Voice", x);
    ENV1(x) = hgroup("[1] Env 1", x);
    FREQ_MOD(x) = hgroup("[2] Freq Mod", x);

    freq = VOICE(vslider("freq", 440., 0., 20000., eps));
    gate = VOICE(button("gate"));
    gain = VOICE(vslider("gain", 0., 0., 1., eps));

    volume = vslider("volume", 0.5, 0., 1., eps);

    attack = ENV1(vslider("[0] Attack", 0.1, 0., .3, eps));
    decay = ENV1(vslider("[1] Decay", 0.1, 0., .3, eps));
    sustain = ENV1(vslider("[2] Sustain", .0, 0., 1., eps));
    release = ENV1(vslider("[3] Release", 0.2, 0., .3, eps));

    cutoff1 = FREQ_MOD(vslider("[0] Mod1", 440., 20., 20000., eps));
    cutoff2 = FREQ_MOD(vslider("[1] Mod2", 15000., 20., 20000., eps));

    // phasor goes from 0 to 1 with frequency f and loops
    phasor(f) = f/ma.SR : (+ : decimalPart) ~ _
    with {
        decimalPart(x) = x-int(x);
    };

    // remap phasor from [0, 1] to [-1, 1]
    sawtooth_cycle(f) = phasor(f) : it.remap(0,1,-1,1);

    // an envelope to control the filter cutoff
    env1 = en.adsr(attack, decay, sustain, release, gate);

    // an envelope to control the volume
    env2 = en.adsr(.01, 0.,.9, 0.1, gate);

    // linearly interpolate between the cutoffs
    cutoff = it.interpolate_linear(env1, cutoff1, cutoff2);

    filter = fi.lowpass(10, cutoff);

    result = freq : sawtooth_cycle : _*gain*env2*volume : filter <: _, _;
};

replace = !,_;
process = ["freq":replace, "gain":replace, "gate":replace -> synth];
"""

with FaustContext():
    box = boxFromDSP(faust_code)
    model_source_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

custom_globals = {}
exec(model_source_code, custom_globals)

MyDSP = custom_globals[module_name]

model = MyDSP(SAMPLE_RATE)

inputs, outputs = model.getNumInputs(), model.getNumOutputs()
assert inputs == 3  # (freq, gain, gate)
assert outputs == 2  # stereo

# print('JAX num inputs: ', inputs, 'num outputs: ', outputs)

########################################################################

jit_inference_fn = jax.jit(model.apply, static_argnums=[2])

# 2 seconds for render length
T = int(SAMPLE_RATE*2.)

# 1 second
hold_length = int(SAMPLE_RATE*1.)

# Play the synth as 440 Hz.
freq = jnp.full((T,), 440.)
# Hold gain and gate at 1.0 for the hold length
gain = (jnp.arange(0, T) < hold_length).astype(jnp.float32)
gate = gain

x = jnp.stack([freq, gain, gate], axis=0)

assert x.shape[0] == 3  # (freq, gain, gate)
key, subkey = random.split(key)
params = model.init({'params': subkey}, x, T)['params']
# print(params)

########################################################################

audio = jit_inference_fn({'params': params}, x, T)
audio = np.array(audio)
# This should sound like the Faust IDE.
# A frequency cutoff rises and falls according to an envelope.

########################################################################

# Now that we've jitted it once, subsequent runs should be faster
# %timeit jit_inference_fn({'params': params}, x, T)

########################################################################

# Create and save data for our soundfile.
wavfile.write("foo.wav", 44100, np.array([0.,1.,0.,-1.]))

faust_code = """
import("stdfaust.lib");
reader = _~+(1);
process = 0,reader:soundfile("param:mySound[url:{'foo.wav'}]",2) : !, !, si.bus(2);
"""

with FaustContext():

    box = boxFromDSP(faust_code)

    # print(f'Inputs: {box.inputs}, Outputs: {box.outputs}')

    module_name = "MyDSP"
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

    custom_globals = {}

    exec(jax_code, custom_globals)

    MyDSP = custom_globals[module_name]
    soundfile_dirs = [str(Path(os.getcwd())), str(Path(os.getcwd())/"MB_Saw")]
    model = MyDSP(sample_rate=SAMPLE_RATE, soundfile_dirs=soundfile_dirs)

    # The generated JAX code will automatically involve calling
    #   self.add_soundfile(state, x, "param:mySound", "{'foo.wav'}", "fSoundfile0")
    # which will subsequently call
    #   self.load_soundfile(self, "foo.wav")
    # Then a learnable jnp.array will be initialized from the audio file "foo.wav".
    # This learnable array is in the params:

    params = model.init({'params': random.PRNGKey(0)}, noises[0], T)['params']
    # print('params:', params)

########################################################################
# Step 1. Define an individual synthesizer voice.

S = 2048  # the length of the wavecycle files we'll create by hand

ramp = np.arange(S)/S

sine = np.sin(ramp*np.pi*2.)

triangle = (1.-2.*abs(abs(np.mod(ramp*2,1.))-.5))*np.where(ramp>0.5, -1, 1)

square = np.where(ramp>0.5, -1, 1)

pwm = np.where(ramp>0.75, -1, 1)

saw = -1+2*ramp

plt.title("Wavetables")
plt.plot(sine, label='sine')
plt.plot(triangle, label='triangle')
plt.plot(square, label='square')
plt.plot(pwm, label='pwm')
plt.plot(saw, label='saw')
plt.legend()
plt.ylabel("Amplitude")
plt.xlabel("Audio Samples")
plt.show()

sine = (sine*np.iinfo(np.int16).max).astype(np.int16)
triangle = (triangle*np.iinfo(np.int16).max).astype(np.int16)
square = (square*np.iinfo(np.int16).max).astype(np.int16)
pwm = (pwm*np.iinfo(np.int16).max).astype(np.int16)
saw = (saw*np.iinfo(np.int16).max).astype(np.int16)

# Save them as wav files.
wavfile.write("sine.wav", 44100, sine)
wavfile.write("triangle.wav", 44100, triangle)
wavfile.write("square.wav", 44100, square)
wavfile.write("pwm.wav", 44100, pwm)
wavfile.write("saw.wav", 44100, saw)

########################################################################
# Sams Stuff

import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Parameters for the Gaussian distribution
sigma = 0.05
mean = 0.0
num_samples = 256  # Number of samples

# Generate an array of x values
x = np.linspace(-1, 1, num_samples)

# Compute the Gaussian PDF values for the x values
pdf_values = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/sigma)**2)

# Normalize the PDF values to fit in the range of int16 audio data (-32768 to 32767)
normalized_pdf_values = np.int16(pdf_values / np.max(np.abs(pdf_values)) * 32767)

# Display the PDF values
plt.plot(x, pdf_values)
plt.title('Gaussian Distribution PDF')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()

# Print the PDF values to confirm
# print("Generated Gaussian Distribution PDF Values:")

# Define the sample rate (arbitrary since we have 256 samples)
sample_rate = 44100  # Standard sample rate for audio files

# Write the values to a .wav file
output_file = 'gaussian_pdf_256_samples.wav'
write(output_file, sample_rate, normalized_pdf_values)

# print(f"WAV file '{output_file}' created with Gaussian PDF of 256 samples.")


########################################################################

# Specify the directory containing the .wav files
directory = "./MB_Saw"

# Get the list of .wav files in the directory without the .wav extension
file_names = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.wav')]

file_names.sort()


num_wave_tables = 16

num_files = len(file_names)


wave_names = []
for i in range(0,num_files,num_files//num_wave_tables):
    wave_names.append(file_names[i])


# Initialize the string to be inserted
insert_string = "wavetables ="

# Append each formatted string to insert_string
for wave in wave_names:
    tmp = f"\n    wavetable(soundfile(\"param:{wave}[url:{{'{wave}.wav'}}]\",1)),"
    insert_string = insert_string + tmp



# Remove the trailing comma and add a semicolon
insert_string = insert_string[:-1] + ";"

# Print the final constructed string
# print(insert_string)

########################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

# Function to plot the waveform of a .wav file
def plot_waveform(wave_file):
    sample_rate, data = read(os.path.join(directory, wave_file))
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(f"Waveform of {wave_file}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

# Plot the waveforms for all the files in the directory
for file in wave_names:
    plot_waveform(file+".wav")

########################################################################


dsp_content = """
import("stdfaust.lib");

// `sf` is a soundfile primitive
wavetable(sf, idx) = it.lagrangeN(N, f_idx, par(i, N + 1, table(i_idx - int(N / 2) + i)))
with {
    N = 3; // lagrange order
    SFC = outputs(sf); // 1 for the length, 1 for the SR, and the remainder are the channels
    S = 0, 0 : sf : ba.selector(0, SFC); // table size
    table(j) = int(ma.modulo(j, S)) : sf(0) : !, !, si.bus(SFC-2);
    f_idx = ma.frac(idx) + int(N / 2);
    i_idx = int(idx);
};
""" + insert_string + """

NUM_TABLES = outputs(wavetables);

// ---------------multiwavetable-----------------------------
// All wavetables are placed evenly apart from each other.
//
// Where:
// * `wt_pos`: wavetable position from [0-1].
// * `ridx`: read index. floating point [0 - (S-1)]
multiwavetable(wt_pos, ridx) = si.vecOp((wavetables_output, coeff), *) :> _
with {
    wavetables_output = ridx <: wavetables;
    // todo: refactor with ef.mixLinearClamp
    coeff = par(i, NUM_TABLES, max(0, 1-abs(i-wt_pos*(NUM_TABLES-1))));
};

S = 2048; // the length of the wavecycles, which we decided elsewhere

wavetable_synth = multiwavetable(wtpos, ridx)*env1*gain
with {
    freq = hslider("freq [style:knob]", 200 , 50  , 1000, .01 );
    gain = hslider("gain [style:knob]", .5  , 0   , 1   , .01 );
    gate = button("gate");

    wtpos = hslider("WT Pos [style:knob]", 0   , 0    , 1   ,  .01);

    ridx = os.hsp_phasor(S, freq, ba.impulsify(gate), 0);
    env1 = en.adsr(.01, 0, 1, .1, gate);
};

replace = !,_;
process = ["freq":replace, "gain":replace, "gate":replace -> wavetable_synth];
"""

# print(dsp_content)

with FaustContext():

    DSP_DIR1 = str(Path(os.getcwd()))
    argv = ['-I', DSP_DIR1]
    box = boxFromDSP(dsp_content, argv)
    module_name = 'FaustVoice'
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

custom_globals = {}
exec(jax_code, custom_globals)  # security risk!
MonoVoice = custom_globals[module_name]

########################################################################
# Step 2. Use jax.linen.vmap to batch several voices, sharing parameters and PRNG among them.

PolyVoice = nn.vmap(MonoVoice, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})

########################################################################
# Step 3. Define the FX module (stereo panner)

dsp_content = """
import("stdfaust.lib");
pan = hslider("pan", 0.5, 0, 1, .01);
process = sp.panner(pan);
"""

with FaustContext():

    box = boxFromDSP(dsp_content,[])
    module_name = 'FaustFX'
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

custom_globals = {}
exec(jax_code, custom_globals)  # security risk!
FaustFX = custom_globals[module_name]

########################################################################
# Step 4. Create an Instrument module that calls the voice batch, sums them, and passes it to the FX module.

soundfile_dirs = [str(Path(os.getcwd())), str(Path(os.getcwd())/"MB_Saw")]

import dataclasses

class PolyInstrument(nn.Module):

    sample_rate: int
    soundfile_dirs: list[str] = dataclasses.field(default_factory=list)

    @nn.compact
    def __call__(self, x, T: int) -> jnp.array:

        # Note that we don't have to think about batches here because we will later call vmap.

        polyvoices = PolyVoice(sample_rate=self.sample_rate, soundfile_dirs=self.soundfile_dirs)

        # audio is shaped (Voices, Channels, Samples)
        audio = polyvoices(x, T)

        # sum all the voices together
        audio = jnp.sum(audio, axis=0)
        # audio is now shaped (Channels, Samples)

        fx = FaustFX(sample_rate=self.sample_rate)

        # apply FX
        audio = fx(audio, T)

        return audio

########################################################################
# Step 5. Use jax.linen.vmap to batch the instrument class without sharing parameters or PRNG.

SAMPLE_RATE = 44100
BatchedInstrument = nn.vmap(PolyInstrument, in_axes=(0, None), variable_axes={'params': 0}, split_rngs={'params': True})
batched_model = BatchedInstrument(SAMPLE_RATE, soundfile_dirs)
jit_batched_inference = jax.jit(partial(batched_model.apply, mutable='intermediates'), static_argnums=[2])

########################################################################
# Step 6. Pass a batched tensor of freq/gain/gate and a batch of parameters to the batched Instrument.

T = int(SAMPLE_RATE*2.)

def pitch_to_hz(pitch):
    return 440.0*(2.0**((pitch - 69)/12.0))

def pitch_to_tensor(pitch, gain, note_dur, total_dur):
    # Return 2D tensor shaped (3, total_dur) where
    # 1st dimension is (freq, gain, gate)
    # and 2nd dimension is audio sample
    freq = pitch_to_hz(pitch)
    tensor = jnp.zeros((3, total_dur))
    tensor = tensor.at[:2, :].set(
        jnp.array([freq, gain]).reshape(2, 1))
    tensor = tensor.at[2, :note_dur].set(jnp.array([1]))

    # We could stack multiple notes on this dimension if we wanted to.
    tensor = jnp.expand_dims(tensor, axis=0)
    return tensor

input_tensor = jnp.stack([
    pitch_to_tensor(60, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(62, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(64, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(65, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(67, 1, int(SAMPLE_RATE*1.), T),
], axis=0)

# print(input_tensor.shape)
BATCH_SIZE = input_tensor.shape[0]
# (Batch, polyphony voices, (freq/gain/gate), samples)

########################################################################

key, subkey = random.split(key)
params = batched_model.init({'params': subkey}, input_tensor, T)['params']
params

########################################################################

params['VmapFaustVoice_0']['_dawdreamer/WT Pos'] = jnp.linspace(-1,1,num=BATCH_SIZE)

audio, mod_vars = jit_batched_inference({'params': params}, input_tensor, T)

########################################################################

# print('audio shape: ', audio.shape)

plt.figure(figsize =(8, 4))

# plt.title("Slice of generated audio")
for i, item in enumerate(audio):
    wavetable_name = ['sine','triangle','square','pwm','saw'][i]

    ax1 = plt.subplot(5, 1, i+1)
    # selected a region where the envelope is at peak sustain
    data = item[0,20000:20400]
    ax1.plot(data, label=wavetable_name)
    if i == 0:
        ax1.set_title("Slices of generated audio")
    plt.tick_params('x', labelbottom = i == len(wavetable_name)-1)

    plt.legend(loc='right')

plt.show()

########################################################################

# Measure speed performance.
# %timeit jit_batched_inference({'params': params}, input_tensor, T)

########################################################################

key, subkey = random.split(key)
wt_pos = random.uniform(subkey, shape=(BATCH_SIZE,), minval=-1, maxval=1)
params['VmapFaustVoice_0']['_dawdreamer/WT Pos'] = wt_pos
wt_pos = np.array(wt_pos)

audio, mod_vars = jit_batched_inference({'params': params}, input_tensor, T)

plt.figure(figsize =(8, 4))

# plt.title("Slice of generated audio")
for i, item in enumerate(audio):
    wavetable_name = f"wtpos={wt_pos[i]:.2f}"

    ax1 = plt.subplot(5, 1, i+1)
    # selected a region where the envelope is at peak sustain
    data = item[0,20000:20400]
    ax1.plot(data, label=wavetable_name)
    if i == 0:
        ax1.set_title("Slices of generated audio")
    plt.tick_params('x', labelbottom = i == len(wavetable_name)-1)

    plt.legend(loc='right')

plt.show()

########################################################################

# print('old input_tensor shape:', input_tensor.shape)

# Cmaj7 chord
chord = jnp.concatenate([
    pitch_to_tensor(60, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(64, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(67, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(71, 1, int(SAMPLE_RATE*1.), T),
], axis=0)

new_tensor = jnp.tile(chord, (5, 1, 1, 1))

# print('new_tensor shape:', new_tensor.shape)

audio, mod_vars = jit_batched_inference({'params': params}, new_tensor, T)

########################################################################
########################################################################