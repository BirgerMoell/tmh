"""
Utils handy for speech processing etc
"""
import librosa
import torch
import soundfile as sf


class ConversionError(Exception):
    pass

def change_sample_rate(audio_path: str, new_sample_rate:int = 16000):
    audio_to_resample, sr = librosa.load(audio_path)
    resampled_audio = librosa.resample(audio_to_resample, sr, new_sample_rate)
    resampled_tensor = torch.tensor([resampled_audio])
    return resampled_tensor

def convert_to_wav(audio_path: str):
    path = audio_path.split("/")
    filename = path[-1].split(".")[0]
    audio, sr = librosa.load(audio_path)
    wav_path = f'{filename}.wav'
    sf.write(wav_path, audio, sr, subtype='PCM_24')
    return wav_path

def ensure_wav(audio_path: str):
    converted = False
    if not audio_path.endswith(".wav"):
        try:
            audio_path = convert_to_wav(audio_path)
            converted = True
        except:
            raise ConversionError(f"Could not convert {audio_path} to wav")
    return audio_path, converted