from lib2to3.pytree import convert
from xdrlib import ConversionError
import torchaudio
import torch
from itertools import groupby
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import pipeline
import librosa

from speechbrain.pretrained import EncoderClassifier
from typing import Any
import soundfile as sf
import os

# from language_files import get_model

class ConversionError(Exception):
    pass

language_dict = {
    "Swedish": "KBLab/wav2vec2-large-voxrex-swedish",
    "English": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "Russian": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "Spanish": "facebook/wav2vec2-large-xlsr-53-spanish",
    "French": "facebook/wav2vec2-large-xlsr-53-french",
    "Persian": "m3hrdadfi/wav2vec2-large-xlsr-persian",
    "Dutch": "facebook/wav2vec2-large-xlsr-53-dutch",
    "Portugese": "facebook/wav2vec2-large-xlsr-53-portuguese",
    "Chinese": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "German": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "Greek": "lighteternal/wav2vec2-large-xlsr-53-greek",
    "Hindi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "Italian": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "Turkish": "cahya/wav2vec2-base-turkish-artificial-cv",
    "Vietnamese": "leduytan93/Fine-Tune-XLSR-Wav2Vec2-Speech2Text-Vietnamese",
    "Catalan": "ccoreilly/wav2vec2-large-100k-voxpopuli-catala",
    "Japanese": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
    "Tamil": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
    "Indonesian": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    "Dhivevi": "shahukareem/wav2vec2-large-xlsr-53-dhivehi",
    "Polish": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "Thai": "sakares/wav2vec2-large-xlsr-thai-demo",
    "Hebrew": "imvladikon/wav2vec2-large-xlsr-53-hebrew",
    "Mongolian": "sammy786/wav2vec2-large-xlsr-mongolian",
    "Czech": "arampacha/wav2vec2-large-xlsr-czech",
    "Icelandic": "m3hrdadfi/wav2vec2-large-xlsr-icelandic",
    "Irish": "jimregan/wav2vec2-large-xlsr-irish-basic",
    "Kinyarwanda": "lucio/wav2vec2-large-xlsr-kinyarwanda",
    "Lithuanian": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "Hungarian": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "Finnish": "aapot/wav2vec2-large-xlsr-53-finnish"
    }

# to do 
# chech language
# enable batch mode
def extract_speaker_embedding(audio_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    signal, fs =torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)
    # print(embeddings)
    return embeddings

def change_sample_rate(audio_path, new_sample_rate=16000):
    audio_to_resample, sr = librosa.load(audio_path)
    resampled_audio = librosa.resample(audio_to_resample, sr, new_sample_rate)
    resampled_tensor = torch.tensor([resampled_audio])
    return resampled_tensor

def classify_emotion(audio_path):
    model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)

    inputs = feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt")

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
    # print(labels)
    return(labels)

def classify_language(audio_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    return(text_lab[0])

def convert_to_wav(audio_path):
    path = audio_path.split("/")
    filename = path[-1].split(".")[0]
    audio, sr = librosa.load(audio_path)
    wav_path = f'{filename}.wav'
    sf.write(wav_path, audio, sr, subtype='PCM_24')
    return wav_path

def transcribe_from_audio_path(audio_path, language='Swedish', check_language=False, classify_emotion=False, model=""):
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

    if check_language:
        language = classify_language(audio_path)
        # print("the language is", language)
        try:
            model_id = language_dict[language]
        except KeyError:
            print("No language model found for %s. Defaulting to KBLab/wav2vec2-large-voxrex-swedish unless another model was specified." %language)
            model_id = "KBLab/wav2vec2-large-voxrex-swedish"
    else:
        try:
            model_id = language_dict[language]
        except KeyError:
            print("No language model found for %s. Defaulting to KBLab/wav2vec2-large-voxrex-swedish unless another model was specified." %language)
            model_id = "KBLab/wav2vec2-large-voxrex-swedish"

    if model:
        model_id = model

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
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

# file_path = "/Users/bmoell/Code/test_tanscribe/sv.wav"
# output = transcribe_from_audio_path(file_path, "English")
# print("the output is", output)
# transcription = "Det visste i varje fall näsan."
# print("the transcription is", transcription)