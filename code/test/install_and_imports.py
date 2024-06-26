# install_and_imports.py
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
from flax.training import train_state
from flax.core.frozen_dict import unfreeze
import optax

from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
from dawdreamer.faust.box import *

from tqdm.notebook import tqdm
import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML, Audio
import IPython.display as ipd
