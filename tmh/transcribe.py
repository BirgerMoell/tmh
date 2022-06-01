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
import numpy as np

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

def get_speech_rate_time_stamps(time_stamps, downsample=320, sample_rate=16000):

    utterances = len(time_stamps[0])
    start_time = time_stamps[0][0]['start_offset']
    end_time = time_stamps[0][utterances-1]['end_offset']
    duration = end_time - start_time
    
    speech_rate = ((duration / utterances) * downsample) / sample_rate
    
    return speech_rate

def calculate_variance(data):
    n = len(data)
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance

def get_speech_rate_variability(time_stamps, type='char', downsample=320, sample_rate=16000 ):
    base = downsample / sample_rate
    token_durations = {}

    for time_stamp in time_stamps[0]:

        start_time = round(time_stamp['start_offset']*base, 2)
        end_time = round(time_stamp['end_offset']*base, 2)
        char = time_stamp[type]
        duration = end_time - start_time

        if char not in token_durations:
            token_durations[char] = []

        token_durations[char].append(duration)

    averages = dict()
    stds = dict()
    variances = dict()

    for token, durations in token_durations.items():
        average = np.sum(durations) / len(durations)
        std = np.std(durations)
        # print("the tokens are", token)
        # print("the durations are", durations)
        # print("the average is", average)
        variance = calculate_variance(durations)
        averages[token] = average
        stds[token] = std
        variances[token] = variance
    
    return averages, stds, variances

def transcribe_from_audio_path(audio_path, language='Swedish', check_language=False, classify_emotion=False, model="", output_word_offsets=False, language_model=False):
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
        logits = model(waveform).logits
    pred_ids = torch.argmax(logits, dim=-1)

    if output_word_offsets:
        outputs = processor.batch_decode(pred_ids, output_word_offsets=output_word_offsets)
        transcription = outputs["text"][0]
        token_time_stamps = outputs[1]
        speech_rate = get_speech_rate_time_stamps(token_time_stamps)
        averages, stds, variances = get_speech_rate_variability(token_time_stamps, type="char")
        word_time_stamps = outputs[2]
        return {
            "transcription": transcription,
            "speech_rate": speech_rate,
            "averages": averages,
            "standard_deviations": stds,
            "variances": variances
        }
    else:
        transcription = processor.batch_decode(pred_ids)
        return transcription[0]


def output_word_offset(pred_ids, processor, output_word_offsets):
    outputs = processor.batch_decode(pred_ids, output_word_offsets=output_word_offsets)
    transcription = outputs["text"][0]
    token_time_stamps = outputs[1]
    speech_rate = get_speech_rate_time_stamps(token_time_stamps)
    averages, stds, variances = get_speech_rate_variability(token_time_stamps, type="char")
    word_time_stamps = outputs[2]
    return {
        "transcription": transcription,
        "speech_rate": speech_rate,
        "averages": averages,
        "standard_deviations": stds,
        "variances": variances
    }


# file_path = "/home/bmoell/tmh/tmh/test.wav"
# output = transcribe_from_audio_path(file_path, "English", output_word_offsets=True)
# print("the output is", output)
# transcription = "Det visste i varje fall näsan."
# print("the transcription is", transcription)