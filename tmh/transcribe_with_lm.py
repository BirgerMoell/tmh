from lib2to3.pytree import convert
from xdrlib import ConversionError
import torchaudio
import torch
from itertools import groupby
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
import librosa

from speechbrain.pretrained import EncoderClassifier
from typing import Any
import soundfile as sf
import os
import numpy as np
from transcribe_with_vad import extract_speak_segments

# from language_files import get_model
def change_sample_rate(audio_path, new_sample_rate=16000):
    audio_to_resample, sr = librosa.load(audio_path)
    resampled_audio = librosa.resample(audio_to_resample, sr, new_sample_rate)
    resampled_tensor = torch.tensor([resampled_audio])
    return resampled_tensor

def convert_to_wav(audio_path):
    path = audio_path.split("/")
    filename = path[-1].split(".")[0]
    audio, sr = librosa.load(audio_path)
    wav_path = f'{filename}.wav'
    sf.write(wav_path, audio, sr, subtype='PCM_24')
    return wav_path

def transcribe_from_audio_path_with_lm(audio_path, model_id="viktor-enzell/wav2vec2-large-voxrex-swedish-4gram"):
    converted = False
    if audio_path[-4:] != ".wav":
        try:
            audio_path = convert_to_wav(audio_path)
            converted = True
        except:
            raise ConversionError(f"Could not convert {audio_path} to wav")
    
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        #resample to 16000 Hz
        waveform = change_sample_rate(audio_path)
        sample_rate = 16000

    if converted:
        os.remove(audio_path)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)

    inputs = processor(waveform[0], sampling_rate=16000, return_tensors='pt', padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    
    transcripts = processor.batch_decode(logits.cpu().numpy()).text
    #print(transcripts)
    return transcripts[0]



def transcribe_from_audio_path_with_lm_vad(audio_path, model_id="viktor-enzell/wav2vec2-large-voxrex-swedish-4gram"):
    converted = False
    if audio_path[-4:] != ".wav":
        try:
            audio_path = convert_to_wav(audio_path)
            converted = True
        except:
            raise ConversionError(f"Could not convert {audio_path} to wav")
    
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        #resample to 16000 Hz
        waveform = change_sample_rate(audio_path)
        sample_rate = 16000

    if converted:
        os.remove(audio_path)

    segments = extract_speak_segments(audio_path)
    transcriptions = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)

    for segment in segments['content']:
        x = waveform[:,int(segment['segment']['start']*sample_rate): int(segment['segment']['end']*sample_rate)]

        inputs = processor(x[0], sampling_rate=16000, return_tensors='pt', padding=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        transcription = processor.batch_decode(logits.cpu().numpy()).text
        full_transcript = {   
            "transcription": transcription[0].encode('utf8').decode(),
            "start": segment['segment']['start'],
            "end": segment['segment']['end']
        }
        #print(transcription)
        transcriptions.append(full_transcript)

    return transcriptions

