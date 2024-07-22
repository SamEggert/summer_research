import time
import os
from pathlib import Path
import jax
import jax.numpy as jnp
from faust_utils import construct_faust_code, convert_faust_to_jax
from audio_utils import pitch_to_tensor, generate_saw_wave, pitch_to_hz, spectrogram_loss
from model import initialize_model
from visualization import plot_spectrograms
from config import SAMPLE_RATE, NUM_STEPS, PITCHES, T

def create_input_tensor(pitches):
    return pitch_to_tensor(pitches[0], 1, T, T)[None, ...]

def generate_target_sound(pitches):
    return generate_saw_wave(pitch_to_hz(pitches[0]), T / SAMPLE_RATE, SAMPLE_RATE)[None, None, :]

def simple_es_step(rng, fitness_fn, mean, std, population_size):
    key, subkey = jax.random.split(rng)
    noise = jax.random.normal(subkey, shape=(population_size,) + mean.shape)
    population = mean + std * noise
    fitnesses = jax.vmap(fitness_fn)(population)
    best_idx = jnp.argmin(fitnesses)
    best_params = population[best_idx]
    return key, best_params, fitnesses[best_idx]

def main():
    parent_dir = Path(__file__).parent.parent
    mb_saw_dir = parent_dir / "MB_Saw"

    print("Loading wavetable files...")
    start_time = time.time()
    file_names = [os.path.splitext(f)[0] for f in os.listdir(mb_saw_dir) if f.endswith('.wav')]
    print(f"Loaded {len(file_names)} wavetable files in {time.time() - start_time:.2f} seconds")

    print("Constructing Faust code...")
    dsp_content = construct_faust_code(mb_saw_dir)

    print("Converting Faust code to JAX...")
    MonoVoice = convert_faust_to_jax(dsp_content)

    print("Initializing model...")
    model, initial_params = initialize_model(MonoVoice)

    print("Creating input tensors...")
    input_tensor = create_input_tensor(PITCHES)

    print("Generating target sound...")
    target_sound = generate_target_sound(PITCHES)

    print("Setting up evolutionary strategy...")
    flat_params, tree_def = jax.tree_util.tree_flatten(initial_params)
    flat_params = jnp.concatenate([p.ravel() for p in flat_params])

    def fitness_function(params):
        reshaped_params = jax.tree_util.tree_unflatten(tree_def, jax.tree_util.tree_leaves(params))
        synth_audio = model.apply({'params': reshaped_params}, input_tensor, T)[0, 0]
        return spectrogram_loss(synth_audio, target_sound[0, 0])

    population_size = 50
    std = 0.1
    rng = jax.random.PRNGKey(0)

    print("Starting evolutionary optimization...")
    for gen in range(NUM_STEPS):
        rng, flat_params, best_fitness = simple_es_step(rng, fitness_function, flat_params, std, population_size)
        if gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness}")

    print("Generating and visualizing results...")
    best_params = jax.tree_util.tree_unflatten(tree_def, jax.tree_util.tree_leaves(flat_params))
    plot_spectrograms(best_params, model.apply, PITCHES[0])

if __name__ == "__main__":
    main()