import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from functools import partial
from scipy.io import wavfile
from faust_to_jax import faust2jax

# Define constants
SAMPLE_RATE = 44100
T = int(SAMPLE_RATE * 1.0)  # 1 second of audio
batch_size = 8

# Define faust code
faust_code = """
import("stdfaust.lib");
cutoff = hslider("cutoff", 440., 20., 20000., .01);
process = fi.lowpass(1, cutoff);
"""

# Function to save audio to WAV file
def save_audio(filename, data, sample_rate=44100):
    # Normalize the data to be within the range of -1 to 1 if needed
    if abs(data).max() > 1.0:
        data /= abs(data).max()
    # Ensure data is a 1D array for mono or a 2D array for stereo
    if data.ndim == 1:
        wavfile.write(filename, sample_rate, (data * 32767).astype(np.int16))
    else:
        wavfile.write(filename, sample_rate, (data.T * 32767).astype(np.int16))

# Initialize models and parameters
hidden_model, jit_hidden = faust2jax(faust_code)
key = random.PRNGKey(42)
key, subkey = random.split(key)

# Generate input noises
input_shape = (batch_size, hidden_model.getNumInputs(), T)
noises = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)
key, subkey = random.split(key)
params = hidden_model.init({'params': subkey}, noises, T)

# Inference
audio, mod_vars = jit_hidden(params, noises, T)

# Save input audio
input_filename = "input_audio.wav"
save_audio(input_filename, np.array(noises[0][0]))
print(f'Input audio saved to {input_filename}')

# Save output audio
output_filename = "output_audio.wav"
save_audio(output_filename, np.array(audio[0][0]))
print(f'Output audio saved to {output_filename}')

# Training example (optional)
def train_step(state, x, y):
    """Train for a single step."""
    def loss_fn(params):
        pred = hidden_model.apply({'params': params}, x, T)
        # L1 time-domain loss
        loss = (jnp.abs(pred - y)).mean()
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Define optimizer
import optax
from flax.training import train_state

learning_rate = 2e-4
momentum = 0.9
tx = optax.sgd(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=hidden_model.apply, params=params, tx=tx)

# Example training loop (optional)
for step in range(100):  # reduce steps for a quick demo
    key, subkey = random.split(key)
    x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)
    y, _ = jit_hidden({'params': params}, x, T)
    state, loss = train_step(state, x, y)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss}")

# Save final output audio after training
final_output_filename = "final_output_audio.wav"
final_audio, _ = jit_hidden(state.params, noises, T)
save_audio(final_output_filename, np.array(final_audio[0][0]))
print(f'Final output audio after training saved to {final_output_filename}')
