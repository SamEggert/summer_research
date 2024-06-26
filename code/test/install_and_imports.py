# install_and_imports.py
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
from dawdreamer.faust.box import *
import scipy
from scipy.io import wavfile
