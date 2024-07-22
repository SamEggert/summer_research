import jax
import optax
from flax.training import train_state
from tqdm import tqdm
import time
from audio_utils import spectrogram_loss
from config import T


@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, x, T)
        loss = spectrogram_loss(pred, y)
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_model(model, params, input_tensor, target_sound, num_steps):
    learning_rate = 1e-2
    tx = optax.adam(learning_rate)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    pbar = tqdm(range(num_steps))
    losses = []

    for step in pbar:
        step_start_time = time.time()
        state, loss = train_step(state, input_tensor, target_sound)
        losses.append(loss)
        step_end_time = time.time()
        pbar.set_description(f"Step {step}, Loss: {loss:.4f}, Step Time: {step_end_time - step_start_time:.2f} seconds")

    return state