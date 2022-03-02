from statistics import variance
from tmh.transcribe import change_sample_rate
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import torch
import torchaudio
import numpy as np

def get_words_and_token_timestamps(audio_path):

    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    waveform, sample_rate = torchaudio.load(audio_path)
    print("the sample rate is", sample_rate)


    if sample_rate != 16000:
        #resample to 16000 Hz
        waveform = change_sample_rate(audio_path)
        sample_rate = 16000

    logits = model(waveform).logits
    pred_ids = torch.argmax(logits, axis=-1)

    outputs = tokenizer.batch_decode(pred_ids, output_word_offsets=True)

    # print("Word time stamps", outputs[0]["word_time_stamps"])
    # print("Token time stamps", outputs[0]["token_time_stamps"])
    transcription = outputs["text"][0]
    token_time_stamps = outputs[1]
    word_time_stamps = outputs[2]

    return (transcription, token_time_stamps, word_time_stamps)


def get_speech_rate_time_stamps(time_stamps, downsample=320, sample_rate=16000):
    

    utterances = len(time_stamps[0])
    start_time = time_stamps[0][0]['start_offset']
    end_time = time_stamps[0][utterances-1]['end_offset']
    duration = end_time - start_time
    
    speech_rate = ((duration / utterances) * downsample) / sample_rate
    
    return speech_rate

def calculate_variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance


def transform_to_seconds(value, downsample=320, sample_rate=16000):
    base = downsample / sample_rate
    value["start_time"] = round(value['start_offset']*base, 2)
    value["end_time"] = round(value['end_offset']*base, 2)
    return value

def add_seconds_to_dict(array):
    seconds_dict = []
    for item in array:
        second_values = transform_to_seconds(item)
        seconds_dict.append(second_values)
    return seconds_dict

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

transcriptions, token_time_stamps, word_time_stamps,  = get_words_and_token_timestamps("/mnt/cloud/data/dementia/media.talkbank.org/dementia/English/Pitt/Control/cookie/295-1.mp3")
speech_rate = get_speech_rate_time_stamps(token_time_stamps)
averages, stds, variances = get_speech_rate_variability(word_time_stamps, type="word")
print("the speech rate is", speech_rate)
print("the averages are", averages)
print("the stds are", stds)
print("the variances are", variances)




