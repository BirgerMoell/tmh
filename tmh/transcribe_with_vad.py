# from vad import extract_speak_segments
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

language_models = {
    "Swedish": "KBLab/wav2vec2-large-voxrex-swedish",
    "English": "jonatasgrosman/wav2vec2-large-xlsr-53-english"
}

from pyannote.audio.pipelines import VoiceActivityDetection

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")

HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}

def extract_speak_segments(audio_path):
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(audio_path)
    # print("extracting speaker segments")
    # print(vad)
    return(vad.for_json())

# def change_sample_rate(audio_path):
#     audio = wave.open(audio_path, mode="wb")
#     audio.setframerate(16000)
#     return audio


def transcribe_from_audio_path_split_on_speech(audio_path, language="Swedish", model=""):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        raise ValueError("sample rate must be 16000")

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
        # print(transcription)

        transcriptions.append({ 
            "transcription": transcription[0],
            "start": segment['segment']['start'],
            "end": segment['segment']['end']
        })
    return transcriptions

# file_path = "/Users/bmoell/Code/test_tanscribe/sv.wav"
# output = transcribe_from_audio_path_split_on_speech(file_path, "Swedish")
# print(output)