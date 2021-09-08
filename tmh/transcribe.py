import torchaudio
import torch
from itertools import groupby
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import pipeline
import librosa
from speechbrain.pretrained import EncoderClassifier
from typing import Any

model_id = "KBLab/wav2vec2-large-voxrex-swedish"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

# to do 
# chech language
# enable batch mode
def extract_speaker_embedding(audio_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    signal, fs =torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)
    print(embeddings)
    return embeddings

def classify_emotion(audio_path):
    model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)

    inputs = feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt")

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
    print(labels)
    return(labels)

def classify_language(audio_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    return(text_lab)

def transcribe_from_audio_path(audio_path, check_language=False, classify_emotion=False):
    waveform, sample_rate = torchaudio.load(audio_path)
    if check_language:
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