"""
Utils handy for speech processing etc
"""
import librosa
import torch
import soundfile as sf
import torchaudio
from scipy.io import wavfile
import noisereduce as nr


class ConversionError(Exception):
    """
    Exception for when conversion fails
    """
    pass


def change_sample_rate(audio_path: str, new_sample_rate: int = 16000):
    """
    Change the sample rate of an audio file. Defaults to 16000 Hz.
    """
    audio_to_resample, sr = librosa.load(audio_path)
    resampled_audio = librosa.resample(audio_to_resample, sr, new_sample_rate)
    resampled_tensor = torch.tensor([resampled_audio])
    return resampled_tensor

def ensure_sample_rate(audio_path: str, sample_rate: int, new_sample_rate: int = 16000):
    """
    Ensure that an audio file is in the specified sample rate. If not, convert it.
    Returns the path to the wav file as well as a boolean specifying if
    the file was converted.
    """
    if sample_rate != new_sample_rate:
        # resample to 16000 Hz
        waveform = change_sample_rate(
            audio_path, new_sample_rate=new_sample_rate)
    else:
        waveform, _ = torchaudio.load(audio_path)
    return waveform

def load_audio(audio_path: str, sample_rate: int = 16000):
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    waveform = ensure_sample_rate(
        audio_path, orig_sample_rate, new_sample_rate=sample_rate)
    return waveform


def convert_to_wav(audio_path: str, output_path: str = None):
    """
    Convert an audio file to wav format. Should be able to handle most file
    types, i.e. mp3, flac, caf, m4a, etc.

    audio_path: path to the audio file
    output_path: path to the output file (ending with .wav)
    """
    path = audio_path.split("/")
    filename = path[-1].split(".")[0]
    audio, sr = librosa.load(audio_path)
    wavpath = f'{filename}.wav'
    if output_path:
        wavpath = f"{output_path}/{wavpath}"
    sf.write(wavpath, audio, sr, subtype='PCM_24')
    print(f"Converted {audio_path} to wav:", wavpath)
    return wavpath


def ensure_wav(audio_path: str, reduce_noise: bool = False):
    """
    Ensure that an audio file is in wav format. If not, convert it.
    Returns the path to the wav file as well as a boolean specifying if
    the file was converted.
    """
    converted = False
    if not audio_path.endswith(".wav"):
        try:
            audio_path = convert_to_wav(audio_path)
            converted = True
        except FileNotFoundError as e:
            raise FileNotFoundError(e)
        except:
            raise ConversionError(f"Could not convert {audio_path} to wav")

    if reduce_noise:
        audio_path = reduce_noise(audio_path)

    return audio_path, converted


def reduce_noise(self, audio_path):
    print(f"Reducing noise for {audio_path}")
    # load data
    rate, data = wavfile.read(audio_path)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(audio_path, rate, reduced_noise)
    return audio_path
