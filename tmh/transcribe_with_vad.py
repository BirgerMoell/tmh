from vad import extract_speak_segments
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

language_models = {
    "Swedish": "KBLab/wav2vec2-large-voxrex-swedish",
    "English": "jonatasgrosman/wav2vec2-large-xlsr-53-english"
}

def transcribe_from_audio_path_split_on_speech(audio_path, language="Swedish", model=""):
    waveform, sample_rate = torchaudio.load(audio_path)
    segments = extract_speak_segments(audio_path)
    transcriptions = []

    model_id = language_models[language]
    if model:
        model_id = model

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    for segment in segments['content']:
        
        x = waveform[:,int(segment['segment']['start']*sample_rate): int(segment['segment']['end']*sample_rate)]
        with torch.no_grad():
            #logits = model(chunk.to("cuda")).logits
            logits = model(x).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)
        print(transcription)

        transcriptions.append({ 
            "transcription": transcription[0],
            "start": segment['segment']['start'],
            "end": segment['segment']['end']
        })
    return transcriptions

# file_path = "/Users/bmoell/Code/test_tanscribe/sv.wav"
# output = transcribe_from_audio_path_split_on_speech(file_path, "Swedish")
# print(output)