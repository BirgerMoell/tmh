import torchaudio
import torch
from itertools import groupby
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from speechbrain.pretrained import EncoderClassifier
from typing import Any

model_id = "KBLab/wav2vec2-large-voxrex-swedish"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

# to do 
# chech language
# enable batch mode

def classify_language(audio_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    print(text_lab)
    return(text_lab)

def transcribe_from_audio_path(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    language = classify_language(audio_path)
    print("the language is", language)
    with torch.no_grad():
        #logits = model(chunk.to("cuda")).logits
        logits = model(waveform).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)
    #get_word_timestamps(transcription[0], pred_ids, chunk, sample_length)
    #print(transcription)
    return transcription[0]


def get_word_timestamps(transcription: str, predicted_ids, input_values, sample_rate) -> Any:
        ##############
    # this is where the logic starts to get the start and end timestamp for each word
    ##############
    words = [w for w in transcription.split(' ') if len(w) > 0]
    predicted_ids = predicted_ids[0].tolist()
    duration_sec = input_values.shape[1] / sample_rate
    ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
    ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]
    split_ids_w_time = [list(group) for k, group
                        in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                        if not k]
    # make sure that there are the same number of id-groups as words. Otherwise something is wrong
    assert len(split_ids_w_time) == len(words), (len(split_ids_w_time), len(words))
    print(transcription)
    print(split_ids_w_time)

# file_path = "/data/asr/asr/slt/wav/t2un3016.wv1.wav"
# output = transcribe_from_audio_path(file_path)
# print("the output is", output)
# transcription = "Det visste i varje fall näsan."
# print("the transcription is", transcription)