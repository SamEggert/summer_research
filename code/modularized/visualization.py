import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from audio_utils import generate_saw_wave, pitch_to_hz, pitch_to_tensor
from config import SAMPLE_RATE


def plot_spectrograms(state, pitch):
    one_second_samples = SAMPLE_RATE

    # Generate target saw wave
    target_audio = generate_saw_wave(pitch_to_hz(pitch), 1, SAMPLE_RATE)
    target_audio_np = np.array(target_audio)

    # Generate synthesized audio
    synth_params = {'params': state.params}
    synth_input = pitch_to_tensor(pitch, 1, one_second_samples, one_second_samples)
    synth_audio = state.apply_fn(synth_params, synth_input[None, ...], one_second_samples)[0, 0]
    synth_audio_np = np.array(synth_audio)

    # Save audio files
    sf.write('target_saw.wav', target_audio_np, SAMPLE_RATE)
    sf.write('synthesized_audio.wav', synth_audio_np, SAMPLE_RATE)

    # Plot spectrograms
    plt.figure(figsize=(14, 6))
    plot_spectrogram(target_audio_np, 'Target Spectrogram', 1)
    plot_spectrogram(synth_audio_np, 'Synthesized Spectrogram', 2)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(audio, title, position):
    plt.subplot(1, 2, position)
    D = librosa.stft(audio, n_fft=4096, hop_length=1024)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title)
    plt.yscale('log')
    plt.ylim(20, SAMPLE_RATE / 2)
    yticks = [20, 50, 100, 200, 440, 1000, 2000, 5000, 10000, 20000]
    plt.yticks(yticks, [str(y) for y in yticks])
    plt.axhline(y=440, color='r', linestyle='--', alpha=0.5)