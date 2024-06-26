# main.py
from install_and_imports import *
from constants_and_utilities import *
from faust_to_jax import faust2jax
from model_inference_and_training import *

def main():
    faust_code = """
    import("stdfaust.lib");
    cutoff = hslider("cutoff", 440., 20., 20000., .01);
    process = fi.lowpass(1, cutoff);
    """
    hidden_model, jit_hidden = faust2jax(faust_code)
    # Execute functions from model_inference_and_training.py

if __name__ == "__main__":
    main()
