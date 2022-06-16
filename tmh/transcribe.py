from lib2to3.pytree import convert
from xdrlib import ConversionError
import torchaudio
import torch
from itertools import groupby
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import pipeline
import librosa
from tmh.utils import ensure_wav, load_audio
from tmh.language_files import get_model
from tmh.transcribe_with_vad import transcribe_from_audio_path_split_on_speech
from tmh.transcribe_with_lm import transcribe_from_audio_path_with_lm, transcribe_from_audio_path_with_lm_vad

from speechbrain.pretrained import EncoderClassifier
from typing import Any
import soundfile as sf
import os
import numpy as np



class ConversionError(Exception):
    pass


# TODO
# check language
# enable batch mode


class TranscribeModel:
    def __init__(self, use_vad=False, use_lm=False, language='Swedish', model_id=None):
        """
        use_vad: use voice activity detection
        use_lm: use language model
        language: language to use
        model_id: the name of a HuggingFace model, overrides `language`

        return: a TranscribeModel object
        """
        self.language = language
        self.use_vad = use_vad
        self.use_lm = use_lm
        self.model_id = model_id if model_id else self.get_model_id()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if use_vad and use_lm:
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id).to(self.device)
            print("Using LM+VAD")
        elif use_vad:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id).to(self.device)
            print("Using VAD")
        elif use_lm:
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id).to(self.device)
            print("Using LM")
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id).to(self.device)
        print("Model was initialized with model_id: {}".format(self.model_id))

    def get_model_id(self):
        if self.use_lm:
            if self.language == "Swedish":
                return "viktor-enzell/wav2vec2-large-voxrex-swedish-4gram"
            else:
                raise ValueError(
                    "Language not supported for LM: {}".format(self.language))
        else:
            return get_model(self.language)

    def transcribe(self, audio_path, classify_emotion=False, output_word_offsets=False, save_to_file=False, reduce_noise=False, output_format: str = "json"):
        """
        audio_path: path to audio file
        classify_emotion: whether to classify emotion or not (not yet implemented) # TODO
        output_word_offsets: whether to output word offsets or not (not yet implemented) # TODO
        save_to_file: whether to save to file or not
        reduce_noise: whether to reduce noise or not
        output_format: the format of the output, either "json" or "text"

        return: transcriptions in the format specified by output_format (default: json)
        """
        if self.use_vad and self.use_lm:
            return transcribe_from_audio_path_with_lm_vad(
                audio_path=audio_path,
                model=self.model,
                processor=self.processor)
        elif self.use_vad:
            return transcribe_from_audio_path_split_on_speech(
                audio_path,
                save_to_file=save_to_file,
                output_format=output_format,
                model=self.model,
                processor=self.processor)
        elif self.use_lm:
            return transcribe_from_audio_path_with_lm(
                audio_path, model=self.model, processor=self.processor).lower()
        else:
            return transcribe_from_audio_path(
                audio_path=audio_path,
                model=self.model,
                processor=self.processor,
                reduce_noise=reduce_noise,
                classify_emotion=classify_emotion,
                output_word_offsets=output_word_offsets)


def extract_speaker_embedding(audio_path):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    signal, fs = torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal)
    # print(embeddings)
    return embeddings

  
def classify_emotion(audio_path):
    audio_path, converted = ensure_wav(audio_path)

    model = HubertForSequenceClassification.from_pretrained(
        "superb/hubert-large-superb-er")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "superb/hubert-large-superb-er")
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)

    inputs = feature_extractor(
        speech, sampling_rate=16000, padding=True, return_tensors="pt")

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
    # print(labels)
    if converted:
        os.remove(audio_path)
    return(labels)


def classify_language(audio_path):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    return(text_lab[0])


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


def get_speech_rate_variability(time_stamps, type='char', downsample=320, sample_rate=16000):
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


def transcribe_from_audio_path(audio_path, model=None, processor=None, model_id=None, language='Swedish', check_language=False, reduce_noise=False, classify_emotion=False, output_word_offsets=False):
    audio_path, converted = ensure_wav(audio_path, reduce_noise=reduce_noise)

    sample_rate = 16000

    if converted:
        os.remove(audio_path)

    if not (model and processor) and not model_id:
        if check_language:
            language = classify_language(audio_path)
        # print("the language is", language)
        model_id = get_model(language)

    if not (model and processor):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    else:
        device = model.device

    transcript = ""
    # Ensure that the sample rate is 16k

    # Stream over 30 seconds chunks rather than load the full file
    stream = librosa.stream(
        audio_path,
        block_length=30,
        frame_length=sample_rate,
        hop_length=sample_rate
    )

    for speech in stream:
        if len(speech.shape) > 1:
            speech = speech[:, 0] + speech[:, 1]

        input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
        logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        transcript += transcription.lower()
        # print(transcription[0])

    if converted:
        os.remove(audio_path)

    return transcript


def output_word_offset(pred_ids, processor, output_word_offsets):
    outputs = processor.batch_decode(
        pred_ids, output_word_offsets=output_word_offsets)
    transcription = outputs["text"][0]
    token_time_stamps = outputs[1]
    speech_rate = get_speech_rate_time_stamps(token_time_stamps)
    averages, stds, variances = get_speech_rate_variability(
        token_time_stamps, type="char")
    word_time_stamps = outputs[2]
    return {
        "transcription": transcription,
        "speech_rate": speech_rate,
        "averages": averages,
        "standard_deviations": stds,
        "variances": variances
    }

