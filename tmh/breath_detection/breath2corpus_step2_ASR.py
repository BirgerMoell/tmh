#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 09:42:34 2019
This script passes the speech segments identified in step 1 through
Google Speech2Text (settings are for English) and creates an updated
textgrid with speech tier.
@author: szekely
"""  
    
from google.cloud import speech_v1p1beta1 as speech
import argparse
import io
import pickle
import os
import csv
import re
import json
import numpy as np
from praatio import tgio
from google.api_core import client_options
import soundfile
import codes
from codes.helpers import list_filenames, annot2textgrid, load_wav


#%% settings
orig_wav_root = './Cormac/denoised_44k/' #location of input wavs
textgrid_root = './Cormac/TG_corrected/' #location of (corrected) TextGrids
output_root = './Cormac/ASR/' #output location

infiles = list(list_filenames(orig_wav_root, extensions='.wav', add_ext=False))
infiles.sort()

sr = 44100


#%% episode settings
epi = 1
episode = 'C'+infiles[epi]
tg_file = episode+'_bc'

output_loc = output_root + episode
if not os.path.exists(output_loc):
    os.makedirs(output_loc) 
if not os.path.exists(output_loc + '/temp'):
    os.makedirs(output_loc + '/temp') 
    
#%%
tg = tgio.openTextgrid(textgrid_root + tg_file + ".TextGrid")
firstTier = tg.tierDict[tg.tierNameList[0]]
tg_start = [entry[0] for entry in firstTier.entryList]
tg_stop = [entry[1] for entry in firstTier.entryList]
tg_label = [entry[2] for entry in firstTier.entryList]
sps = [i for i, x in enumerate(tg_label) if x == 'sp']
# test if speech segments are correctly placed (not at start or end, not next to each other)
test_sp = [sps[i+1]-sps[max(0,i)] for i, x in enumerate(sps[1:])]
print(f"first element is SP: {sps[0]==0}")
print(f"last element is SP: {sps[-1]==len(tg_label)-1}")
print(f"consequtive SPs: {[sps[i] for i, x in enumerate(test_sp) if x==1]}")

#%%
y = load_wav(orig_wav_root+infiles[epi]+'.wav', sr=sr)
wav_out = np.asarray(y[1])
for i in range(len(sps)):
    wav_temp = wav_out[int(sr*tg_start[sps[i]-1]):int(sr*tg_stop[sps[i]+1])]
    wav_name = f"{episode}_{str(i).zfill(4)}.wav"
    soundfile.write(output_loc + '/temp/' + wav_name, wav_temp, sr, subtype='PCM_16')
    
asr_files = list(list_filenames(output_loc + '/temp', extensions='.wav', add_ext=False))
asr_files.sort()
    

#%% set Google Cloud speech2text credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./codes/transcribespeech-312319-d025b453aaeb.json"
    
#%% transcribe list
results = [None]*len(sps)
longresults = [None]*len(sps)
client_ops = client_options.ClientOptions(api_endpoint="eu-speech.googleapis.com")

#%% run ASR on each breath group
for i in range(len(sps)):
    client = speech.SpeechClient(client_options=client_ops)
    
    metadata = speech.types.RecognitionMetadata()
    metadata.interaction_type = (
        speech.RecognitionMetadata.InteractionType.PRESENTATION)
    metadata.microphone_distance = (
        speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD)
    metadata.recording_device_type = (
        speech.RecognitionMetadata.RecordingDeviceType.OTHER_INDOOR_DEVICE)
    metadata.recording_device_type = (
        speech.RecognitionMetadata.OriginalMediaType.AUDIO)
    
    with io.open( f"{output_loc}/temp/{episode}_{str(i).zfill(4)}.wav", 'rb') as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US',
        enable_automatic_punctuation=False,
        enable_word_time_offsets=True,
        use_enhanced=True,
        model='video',
        #speech_contexts=[speech.types.SpeechContext(
            #phrases=['ThinkComputers', 'Podcast', 'Bob', 'Derrick', 'Intel', 'AMD', 'ASUS', 'envy ME', 'Corsair', 'Nvidia', 'GPU', 'CPU', 'RGB', 'Fortnite']
        #)],        
        metadata=metadata)
    
    response = client.recognize(config=config, audio=audio)
    if len(response.results) == 1:
        results[i] = response.results[0].alternatives[0]
    elif len(response.results) == 2:
        results[i] = response.results[0].alternatives[0]
        longresults[i] = response.results[1].alternatives[0]        
    elif len(response.results) > 2:        
        results[i] = response.results[0].alternatives[0]
        longresults[i] = response.results[1].alternatives[0]        
        print(f"multiple responses: {i}")

#%% extract transcript
transcript = [None]*len(sps)
for i in range(len(results)):
    try:
        transcript[i] = results[i].transcript
    except:
        print('empty transcript', i)
        
for j in range(len(longresults)):
    if longresults[j]:
        print('second line', j)
        transcript[j] += (' ' + longresults[j].transcript)

transcript2 = [None]*len(sps)
for k in range(len(transcript)):
    if transcript[k]:
        transcript2[k] = re.sub(r"(?<=\w)([A-Z])", r" \1", transcript[k])
        transcript2[k] = re.sub(r"([a-z])\-([a-z])", r"\1 \2", transcript2[k] , 0, re.IGNORECASE)

#%% save results
tg_asr = list(tg_label)
for i in range(len(sps)):
    if transcript2[i]:
        tg_asr[sps[i]] = transcript2[i]

annotTier = tgio.IntervalTier('labels', [], 0, pairedWav=orig_wav_root+infiles[epi]+'.wav')
asrTier = tgio.IntervalTier('transcript', [], 0, pairedWav=orig_wav_root+infiles[epi]+'.wav')
        
tg2 = tgio.Textgrid()
tg2.addTier(annotTier)
tg2.addTier(asrTier)
for i in range(len(tg_start)):
    annotTier.insertEntry((tg_start[i], tg_stop[i], tg_label[i]), warnFlag=True, collisionCode='replace')
    asrTier.insertEntry((tg_start[i], tg_stop[i], tg_asr[i]), warnFlag=True, collisionCode='replace')

tg2.save(output_root + episode + ".TextGrid")