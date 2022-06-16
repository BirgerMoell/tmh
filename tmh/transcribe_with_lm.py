import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
import os
from tmh.transcribe_with_vad import extract_speak_segments
from tmh.utils import load_audio, ensure_wav

# from language_files import get_model


def transcribe_from_audio_path_with_lm(audio_path, model_id="viktor-enzell/wav2vec2-large-voxrex-swedish-4gram", model=None, processor=None):
    audio_path, converted = ensure_wav(audio_path)

    sample_rate = 16000
    waveform = load_audio(audio_path, sample_rate)

    if converted:
        os.remove(audio_path)

    if not (model and processor):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
    else:
        device = model.device

    inputs = processor(waveform[0], sampling_rate=16000,
                       return_tensors='pt', padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    transcripts = processor.batch_decode(logits.cpu().numpy()).text
    # print(transcripts)
    return transcripts[0]


def transcribe_from_audio_path_with_lm_vad(audio_path, model_id="viktor-enzell/wav2vec2-large-voxrex-swedish-4gram", model=None, processor=None):
    audio_path, converted = ensure_wav(audio_path)

    sample_rate = 16000
    waveform = load_audio(audio_path, sample_rate)

    segments = extract_speak_segments(audio_path)
    transcriptions = []

    if converted:
        os.remove(audio_path)

    if not (model and processor):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
    else:
        device = model.device

    for segment in segments['content']:
        x = waveform[:, int(segment['segment']['start']*sample_rate)
                            : int(segment['segment']['end']*sample_rate)]

        inputs = processor(x[0], sampling_rate=sample_rate,
                           return_tensors='pt', padding=True).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        transcription = processor.batch_decode(logits.cpu().numpy()).text
        full_transcript = {
            "transcription": transcription[0].encode('utf8').decode().lower(),
            "start": segment['segment']['start'],
            "end": segment['segment']['end']
        }
        # print(transcription)
        transcriptions.append(full_transcript)

    if converted:
        os.remove(audio_path)

    return transcriptions
