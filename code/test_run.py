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

def show_audio(data, autoplay=False):
    if abs(data).max() > 1.:
        data /= abs(data).max()
    return Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay)

def faust2jax(faust_code: str):
    module_name = "MyDSP"
    with FaustContext():
        box = boxFromDSP(faust_code)
        jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

    custom_globals = {}
    exec(jax_code, custom_globals)  # security risk!

    MyDSP = custom_globals[module_name]
    MyDSP = nn.vmap(MyDSP, in_axes=(0, None), variable_axes={'params': None, 'intermediates': 0}, split_rngs={'params': False})

    model_batch = MyDSP(sample_rate=SAMPLE_RATE)
    jit_inference_fn = jax.jit(partial(model_batch.apply, mutable='intermediates'), static_argnums=[2])
    return model_batch, jit_inference_fn

faust_code = """
import("stdfaust.lib");
cutoff = hslider("cutoff", 440., 20., 20000., .01);
process = fi.lowpass(1, cutoff);
"""
hidden_model, jit_hidden = faust2jax(faust_code)

# Training example
T = int(SAMPLE_RATE * 1.0)  # 1 second of audio
batch_size = 8
input_shape = (batch_size, hidden_model.getNumInputs(), T)

key = random.PRNGKey(42)
key, subkey = random.split(key)
noises = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)

key, subkey = random.split(key)
params = hidden_model.init({'params': subkey}, noises, T)

print('normalized params: ', params)
audio, mod_vars = jit_hidden(params, noises, T)
print('un-normalized params: ', mod_vars['intermediates'])

print('input audio (LOUD!!):')
show_audio(np.array(noises[0]))

print('output audio (less loud):')
show_audio(np.array(audio[0]))

# Example of using soundfile with JAX
wavfile.write("foo.wav", 44100, np.array([0., 1., 0., -1.]))

faust_code = """
import("stdfaust.lib");
reader = _~+(1);
process = 0,reader:soundfile("param:mySound[url:{'foo.wav'}]",2) : !, !, si.bus(2);
"""
with FaustContext():
    box = boxFromDSP(faust_code)
    print(f'Inputs: {box.inputs}, Outputs: {box.outputs}')
    jax_code = boxToSource(box, 'jax', "MyDSP", ['-a', 'jax/minimal.py'])

custom_globals = {}
exec(jax_code, custom_globals)
MyDSP = custom_globals["MyDSP"]
soundfile_dirs = [str(Path(os.getcwd())), str(Path(os.getcwd())/"MB_Saw")]
model = MyDSP(sample_rate=SAMPLE_RATE, soundfile_dirs=soundfile_dirs)

params = model.init({'params': random.PRNGKey(0)}, noises[0], T)['params']
print('params:', params)

# Create a differentiable polyphonic wavetable synthesizer with FX
directory = "MB_Saw"
file_names = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.wav')]
file_names.sort()
num_wave_tables = 16
wave_names = [file_names[i] for i in range(0, len(file_names), len(file_names)//num_wave_tables)]

insert_string = "wavetables ="
for wave in wave_names:
    tmp = f"\n    wavetable(soundfile(\"param:{wave}[url:{{'{wave}.wav'}}]\",1)),"
    insert_string += tmp
insert_string = insert_string[:-1] + ";"

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
    // todo: refactor with ef.mixLinearClamp
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

with FaustContext():
    DSP_DIR1 = str(Path(os.getcwd()))
    argv = ['-I', DSP_DIR1]
    box = boxFromDSP(dsp_content, argv)
    module_name = 'FaustVoice'
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

custom_globals = {}
exec(jax_code, custom_globals)
MonoVoice = custom_globals[module_name]

PolyVoice = nn.vmap(MonoVoice, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})

dsp_content = """
import("stdfaust.lib");
pan = hslider("pan", 0.5, 0, 1, .01);
process = sp.panner(pan);
"""
with FaustContext():
    box = boxFromDSP(dsp_content, [])
    module_name = 'FaustFX'
    jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

custom_globals = {}
exec(jax_code, custom_globals)
FaustFX = custom_globals[module_name]

import dataclasses

class PolyInstrument(nn.Module):
    sample_rate: int
    soundfile_dirs: list[str] = dataclasses.field(default_factory=list)

    @nn.compact
    def __call__(self, x, T: int) -> jnp.array:
        polyvoices = PolyVoice(sample_rate=self.sample_rate, soundfile_dirs=self.soundfile_dirs)
        audio = polyvoices(x, T)
        audio = jnp.sum(audio, axis=0)
        fx = FaustFX(sample_rate=self.sample_rate)
        audio = fx(audio, T)
        return audio

BatchedInstrument = nn.vmap(PolyInstrument, in_axes=(0, None), variable_axes={'params': 0}, split_rngs={'params': True})
batched_model = BatchedInstrument(SAMPLE_RATE, soundfile_dirs)
jit_batched_inference = jax.jit(partial(batched_model.apply, mutable='intermediates'), static_argnums=[2])

def pitch_to_hz(pitch):
    return 440.0*(2.0**((pitch - 69)/12.0))

def pitch_to_tensor(pitch, gain, note_dur, total_dur):
    freq = pitch_to_hz(pitch)
    tensor = jnp.zeros((3, total_dur))
    tensor = tensor.at[:2, :].set(jnp.array([freq, gain]).reshape(2, 1))
    tensor = tensor.at[2, :note_dur].set(jnp.array([1]))
    tensor = jnp.expand_dims(tensor, axis=0)
    return tensor

input_tensor = jnp.stack([
    pitch_to_tensor(60, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(62, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(64, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(65, 1, int(SAMPLE_RATE*1.), T),
    pitch_to_tensor(67, 1, int(SAMPLE_RATE*1.), T),
], axis=0)

BATCH_SIZE = input_tensor.shape[0]
key, subkey = random.split(key)
params = batched_model.init({'params': subkey}, input_tensor, T)['params']
params['VmapFaustVoice_0']['_dawdreamer/WT Pos'] = jnp.linspace(-1, 1, num=BATCH_SIZE)

audio, mod_vars = jit_batched_inference({'params': params}, input_tensor, T)

for item in audio:
    show_audio(item)

plt.figure(figsize=(8, 4))
for i, item in enumerate(audio):
    wavetable_name = ['sine','triangle','square','pwm','saw'][i]
    ax1 = plt.subplot(5, 1, i+1)
    data = item[0, 20000:20400]
    ax1.plot(data, label=wavetable_name)
    if i == 0:
        ax1.set_title("Slices of generated audio")
    plt.tick_params('x', labelbottom = i == len(wavetable_name)-1)
    plt.legend(loc='right')
plt.show()
