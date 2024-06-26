# constants_and_utilities.py
import IPython.display as ipd
from IPython.display import Audio

SAMPLE_RATE = 44100

def show_audio(data, autoplay=False):
    if abs(data).max() > 1.:
        data /= abs(data).max()
    ipd.display(Audio(data=data, rate=SAMPLE_RATE, normalize=False, autoplay=autoplay))
