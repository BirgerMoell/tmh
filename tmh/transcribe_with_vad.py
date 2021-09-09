from vad import extract_silences
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

model_id = "KBLab/wav2vec2-large-voxrex-swedish"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)


def transcribe_from_audio_path_split_on_pause(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    silences = extract_silences(waveform, sample_rate)
    with torch.no_grad():
        #logits = model(chunk.to("cuda")).logits
        logits = model(waveform).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)
    #get_word_timestamps(transcription[0], pred_ids, chunk, sample_length)
    #print(transcription)
    return transcription[0]
