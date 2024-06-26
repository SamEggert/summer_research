from dawdreamer.faust import createLibContext, destroyLibContext, FaustContext
from dawdreamer.faust.box import boxFromDSP, boxToSource
from flax import linen as nn
import jax
import jax.numpy as jnp
from functools import partial

SAMPLE_RATE = 44100

def faust2jax(faust_code: str):
    """
    Convert faust code into a batched JAX model and a single-item inference function.

    Inputs:
    * faust_code: string of faust code.
    """

    module_name = "MyDSP"
    with FaustContext():
        box = boxFromDSP(faust_code)
        jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

    custom_globals = {}

    exec(jax_code, custom_globals)  # security risk!

    MyDSP = custom_globals[module_name]
    # MyDSP is now a class definition which can be instantiated or subclassed.

    """
    In our vmap code, we specify `in_axes=(0, None)` to batch along `x` but not along `T`.
    In other words, the `x` argument will be a batch, but our `T` will remain a simple integer (not a batch of integers)
    We choose `variable_axes={'params': None, 'intermediates': 0}` in order to share parameters among the batch and continue to sow intermediate vars
    We choose `split_rngs={'params': False}` to use the same random number generator for each item in the batch.
    """
    MyDSP = nn.vmap(MyDSP, in_axes=(0, None), variable_axes={'params': None, 'intermediates': 0}, split_rngs={'params': False})

    # Now we can create a model that handles batches of input.
    model_batch = MyDSP(sample_rate=SAMPLE_RATE)

    # let's jit compile the model's `apply` method
    jit_inference_fn = jax.jit(partial(model_batch.apply, mutable='intermediates'), static_argnums=[2])

    # We jitted the model's "apply" function, which is of the form `apply(params, x, T)`.
    # T (the number of samples of the output) is a constant, so we specified static_argnums=[2].

    # We specify mutable='intermediates' to access intermediate variables, which are
    # human-interpretable.
    # Our architecture file normalized all of the parameters to be between -1 and 1.
    # During a forward pass, the parameters are remapped to their original ranges
    # and stored as intermediate variables via the `sow` method.

    return model_batch, jit_inference_fn
