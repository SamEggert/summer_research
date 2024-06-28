import os
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
from flax.core.frozen_dict import unfreeze
import optax

from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
from dawdreamer.faust.box import boxFromDSP, boxToSource

import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import Audio, HTML


SAMPLE_RATE = 44100

# PUT FAUST CODE HERE
faust_code = ""

def faust2jax(faust_code: str):
    """
    Convert faust code into a batched JAX model and a single-item inference function.

    Inputs:
    * faust_code: string of faust code.
    """

    module_name = "MyDSP"
    print("REACHED 0")
    with FaustContext():

      box = boxFromDSP(faust_code)

      jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])
    print("REACHED 1")

    custom_globals = {}
    print("REACHED 2")

    exec(jax_code, custom_globals)  # security risk!
    print("REACHED 3")

    MyDSP = custom_globals[module_name]
    print("REACHED 4")
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

########################################################################
# My code for parsing the wavetables into faust code
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

print(input_tensor.shape)
BATCH_SIZE = input_tensor.shape[0]
# (Batch, polyphony voices, (freq/gain/gate), samples)


########################################################################
key, subkey = random.split(key)
params = batched_model.init({'params': subkey}, input_tensor, T)['params']
# params
########################################################################
params['VmapFaustVoice_0']['_dawdreamer/WT Pos'] = jnp.linspace(-1,1,num=BATCH_SIZE)

audio, mod_vars = jit_batched_inference({'params': params}, input_tensor, T)
########################################################################



