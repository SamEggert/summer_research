import time
import os
from pathlib import Path
from faust_utils import construct_faust_code, convert_faust_to_jax
from audio_utils import pitch_to_tensor, generate_saw_wave, pitch_to_hz
from model import initialize_model
from training import train_model
from visualization import plot_spectrograms
from config import SAMPLE_RATE, NUM_STEPS, PITCHES, T

def create_input_tensor(pitches):
    return pitch_to_tensor(pitches[0], 1, T, T)[None, ...]

def generate_target_sound(pitches):
    return generate_saw_wave(pitch_to_hz(pitches[0]), T / SAMPLE_RATE, SAMPLE_RATE)[None, None, :]

def main():
    # Get the path to the directory containing the 'modularized' folder
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
    model, params = initialize_model(MonoVoice)

    print("Creating input tensors...")
    input_tensor = create_input_tensor(PITCHES)

    print("Generating target sound...")
    target_sound = generate_target_sound(PITCHES)

    print("Starting training loop...")
    trained_state = train_model(model, params, input_tensor, target_sound, NUM_STEPS)

    print("Generating and visualizing results...")
    plot_spectrograms(trained_state, PITCHES[0])

if __name__ == "__main__":
    main()