# faust_to_jax.py
from install_and_imports import *

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
    MyDSP = nn.vmap(MyDSP, in_axes=(0, None), variable_axes={'params': None, 'intermediates': 0}, split_rngs={'params': False})
    model_batch = MyDSP(sample_rate=SAMPLE_RATE)
    jit_inference_fn = jax.jit(partial(model_batch.apply, mutable='intermediates'), static_argnums=[2])
    return model_batch, jit_inference_fn
