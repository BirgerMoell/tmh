from lib2to3.pytree import convert
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
from tmh.transcribe_with_vad import extract_speak_segments
from utils import change_sample_rate, ensure_wav

# from language_files import get_model

def transcribe_from_audio_path_with_lm(audio_path, model_id="viktor-enzell/wav2vec2-large-voxrex-swedish-4gram"):
    audio_path, converted = ensure_wav(audio_path)
    
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
    audio_path, converted = ensure_wav(audio_path)
    
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

