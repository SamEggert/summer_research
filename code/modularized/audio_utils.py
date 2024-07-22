import jax.numpy as jnp
import jax.scipy.signal as jss
from config import SAMPLE_RATE

def pitch_to_hz(pitch):
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

def pitch_to_tensor(pitch, gain, note_dur, total_dur):
    freq = pitch_to_hz(pitch)
    tensor = jnp.zeros((3, total_dur))
    tensor = tensor.at[:2, :].set(jnp.array([freq, gain]).reshape(2, 1))
    tensor = tensor.at[2, :note_dur].set(jnp.array([1]))
    return tensor

def generate_saw_wave(frequency, duration, sample_rate):
    t = jnp.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    saw_wave = 2 * (t * frequency - jnp.floor(t * frequency + 0.5))
    return saw_wave

def spectrogram(x, fft_size=2048, hop_length=512):
    if x.ndim == 1:
        x = x[jnp.newaxis, :]
    f, t, Zxx = jss.stft(x, fs=SAMPLE_RATE, nperseg=fft_size, noverlap=fft_size-hop_length, padded=False, boundary=None)
    return jnp.abs(Zxx)

def spectrogram_loss(pred, target):
    pred_spec = spectrogram(pred)
    target_spec = spectrogram(target)
    return jnp.mean(jnp.abs(pred_spec - target_spec))