from flax import linen as nn
from functools import partial
import jax
import os
from pathlib import Path
from config import SAMPLE_RATE, T


def initialize_model(MonoVoice):
    PolyVoice = nn.vmap(MonoVoice, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})

    # Get the path to the current directory and the parent directory
    current_dir = Path(os.getcwd())
    parent_dir = current_dir.parent

    soundfile_dirs = [
        str(current_dir),
        str(parent_dir / "MB_Saw"),
        '/mnt/c/Users/braun/GitHub/sam-eggert-summer_research/code/MB_Saw'  # Keep this as a fallback
    ]

    batched_model = PolyVoice(SAMPLE_RATE, soundfile_dirs=soundfile_dirs)
    jit_batched_inference = jax.jit(partial(batched_model.apply, mutable='intermediates'), static_argnums=[2])

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # Create a dummy input tensor for initialization
    dummy_input = jax.numpy.zeros((1, 3, T))

    params = batched_model.init({'params': subkey}, dummy_input, T)['params']

    return batched_model, params