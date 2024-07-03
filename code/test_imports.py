import jax
import jax.numpy as jnp
from flax import linen as nn
import flax

print("JAX version:", jax.__version__)
print("Flax version:", flax.__version__)

# Simple test to ensure JAX is working
x = jnp.array([1, 2, 3])
y = jnp.sum(x)
print("JAX test result:", y)

# Simple test to ensure Flax is working
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=1)(x)

model = SimpleModel()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))
print("Flax test successful")
