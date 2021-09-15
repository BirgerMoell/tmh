from tensorflow import keras

import os
import numpy as np
from matplotlib import pyplot as plt

from scipy import signal
from scipy import io
from scipy.io import wavfile
import soundfile as sf

import librosa
import librosa.display

import pandas as pd


def load_wav(fn, sr=None, normalize=True):
    if fn == '': # ignore empty filenames
        print('filename missing')
        return None
    fs, audio = wavfile.read(fn)
    audio = audio.astype(np.float32)
    duration = np.shape(audio)[0]
    if duration == 0: # ignore zero-length samples
        print('sample has no length')
        return None
    if sr != fs and sr != None:
        audio = librosa.resample(audio, fs, sr)
        fs = sr
    max_val = np.abs(audio).max()
    if max_val == 0: # ignore completely silent sounds
        print('silent sample')
        return None
    if normalize:
        audio = audio / max_val
    #audio = audio.astype(np.int16)
    return (fn, audio, duration, fs)


model = keras.models.load_model('./modelMix4.h5')
print(model)

x = load_wav("../sample_input/1_1.wav")

# figure out how to make it five dimensional

import pdb
pdb.set_trace()
print(x)
output = model(x[1])
print(output)


