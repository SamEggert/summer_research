from flax import linen as nn
from functools import partial
import jax
import jax.numpy as jnp
import os
from pathlib import Path
from config import SAMPLE_RATE, T


def initialize_model(MonoVoice):
    # Get the path to the directory containing the 'modularized' folder
    parent_dir = Path(__file__).parent.parent
    mb_saw_dir = parent_dir / "MB_Saw"

    soundfile_dirs = [
        str(Path(os.getcwd())),
        str(mb_saw_dir),
        '/mnt/c/Users/braun/GitHub/sam-eggert-summer_research/code/MB_Saw'
    ]

    PolyVoice = nn.vmap(MonoVoice, in_axes=(0, None), variable_axes={'params': None}, split_rngs={'params': False})
    batched_model = PolyVoice(SAMPLE_RATE, soundfile_dirs=soundfile_dirs)

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # Create a dummy input tensor for initialization
    dummy_input = jnp.zeros((1, 3, T))

    params = batched_model.init({'params': subkey}, dummy_input, T)['params']

    # Ensure all parameters have at least 1 dimension
    def ensure_1d(param):
        if param.ndim == 0:
            return param.reshape(1)
        return param

    params = jax.tree_map(ensure_1d, params)

    return batched_model, params