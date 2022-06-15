"""
Utils handy for speech processing etc
"""
import librosa
import torch
import soundfile as sf
import torchaudio


class ConversionError(Exception):
    """
    Exception for when conversion fails
    """
    pass

def change_sample_rate(audio_path: str, new_sample_rate:int = 16000):
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
        #resample to 16000 Hz
        waveform = change_sample_rate(audio_path, new_sample_rate=new_sample_rate)
    else:
        waveform, _ = torchaudio.load(audio_path)
    return waveform

def load_audio(audio_path: str, sample_rate: int = 16000):
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    waveform = ensure_sample_rate(audio_path, orig_sample_rate, new_sample_rate=sample_rate)
    return waveform

def convert_to_wav(audio_path: str):
    """
    Convert an audio file to wav format. Should be able to handle most file
    types, i.e. mp3, flac, caf, m4a, etc.
    """
    path = audio_path.split("/")
    filename = path[-1].split(".")[0]
    audio, sr = librosa.load(audio_path)
    wav_path = f'{filename}.wav'
    sf.write(wav_path, audio, sr, subtype='PCM_24')
    return wav_path

def ensure_wav(audio_path: str):
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
        except:
            raise ConversionError(f"Could not convert {audio_path} to wav")
    return audio_path, converted