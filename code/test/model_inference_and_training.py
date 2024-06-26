# model_inference_and_training.py
from install_and_imports import *
from faust_to_jax import faust2jax

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

init_cutoff = 10000
faust_code = f"""
import("stdfaust.lib");
cutoff = hslider("cutoff", {init_cutoff}, 20., 20000., .01);
process = fi.lowpass(1, cutoff);
"""
train_model, jit_train_inference = faust2jax(faust_code)

input_shape = (batch_size, train_model.getNumInputs(), T)
key, subkey = random.split(key)
noises = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)
key, subkey = random.split(key)
hidden_params = hidden_model.init({'params': subkey}, jnp.zeros_like(noises), T)['params']
key, subkey = random.split(key)
train_params = train_model.init({'params': subkey}, jnp.zeros_like(noises), T)['params']

print('hidden params:', hidden_params)
print('train params:', train_params)

learning_rate = 2e-4
momentum = 0.9
tx = optax.sgd(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=train_model.apply, params=train_params, tx=tx)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        pred = train_model.apply({'params': params}, x, T)
        loss = (jnp.abs(pred - y)).mean()
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

losses, cutoffs = [], []
train_steps = 3000
train_steps_per_eval = 100
pbar = tqdm(range(train_steps))

for n in pbar:
    key, subkey = random.split(key)
    x = random.uniform(subkey, shape=input_shape, minval=-1, maxval=1)
    y, _ = jit_hidden({'params': hidden_params}, x, T)
    state, loss = train_step(state, x, y)
    if n % train_steps_per_eval == 0:
        audio, mod_vars = jit_train_inference({'params': state.params}, noises, T)
        cutoff = np.array(mod_vars['intermediates']['dawdreamer/cutoff']).mean()
        losses.append(loss)
        cutoffs.append(cutoff)
        pbar.set_description(f"cutoff: {cutoff}")

print('Done!')

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
