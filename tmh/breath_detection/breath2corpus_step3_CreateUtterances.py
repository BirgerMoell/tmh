#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:02:37 2021
This script creates overlapping multi breath-group utterances and wav files
for training TTS from the transcription file created in step 2.
@author: szekely
"""
import os
import codes
from codes.helpers import list_filenames, annot2textgrid, load_wav
from praatio import tgio
import pandas as pd
import numpy as np
import soundfile

#%% settings
orig_wav_root = './Cormac/denoised_44k/' #location of input wavs
textgrid_root = './Cormac/ASR/' #location of ASR TextGrids
output_root = './Cormac/corpus/' #output location

infiles = list(list_filenames(orig_wav_root, extensions='.wav', add_ext=False))
infiles.sort()

sr = 44100

# corpus creation settings
max_utt = 11 # maximum utterance length (seconds)
min_utt = 1.0 # minimum utterance length (seconds)
trail = 0.025 # silence before and after utterance (seconds)
margin = 0.025 # additional margin at end of utterance to complete plosives
max_sil = 2 # don't include silences greater than this at start of utterance

#%% episode settings
epi = 1
episode = 'C'+infiles[epi]
tg_file = episode

output_loc = output_root + episode + '/'
if not os.path.exists(output_loc):
    os.makedirs(output_loc) 

#%% load textgrid
tg = tgio.openTextgrid(textgrid_root + tg_file + ".TextGrid")
firstTier = tg.tierDict[tg.tierNameList[0]]
ASRTier = tg.tierDict[tg.tierNameList[1]]

tg_start = [entry[0] for entry in firstTier.entryList]
tg_stop = [entry[1] for entry in firstTier.entryList]
tg_label = [entry[2] for entry in firstTier.entryList]
tg_text = [entry[2] for entry in ASRTier.entryList]
sps = [i for i, x in enumerate(tg_label) if x == 'sp'] # speech segments
sb = [i for i, x in enumerate(tg_label) if x in ['sil', 'b']] # silence and breaths

# exclude long silences
tg_dur = [x[1] - x[0] for x in zip(tg_start, tg_stop)]
sb2 = [x for i, x in enumerate(sb) if tg_label == "b" or tg_dur[x] <= max_sil]

#%% create dataframe for output
speechend = [x for i, x in enumerate(tg_stop) if i in sps]
outfiles = [f"{episode}_{x:03d}" for i, x in enumerate(sps)]
df = pd.DataFrame.from_records(zip(outfiles, sps), columns=["file", "location"])
df['startloc'] = 0
df['starttime'] = 0.
df['endloc'] = 0
df['endtime'] = 0.
df['include'] = 1
df['duration'] = 0.
df['text'] = ''

#%% create multi-breathgroups
for i in range(len(df)):
    if sps[i]-2 in sb2 and sps[i]-1 in sb2:
        df['startloc'][i] = sps[i]-2
    elif sps[i]-1 in sb2:
        df['startloc'][i] = sps[i]-1
    else:
        df['startloc'][i] = sps[i]
    df['starttime'][i] = tg_start[df['startloc'][i]]
    maxtime = df['starttime'][i]+max_utt
    df['endloc'][i] = sps[next((i for i, x in enumerate(speechend) if x > 
                               maxtime), 0)-1]
    df['endtime'][i] = tg_stop[df['endloc'][i]]
    df['duration'][i] = df['endtime'][i] - df['starttime'][i]
    if i>0 and df['endloc'][i] == df['endloc'][i-1]:
        df['include'][i] = 0
    elif df['duration'][i] < min_utt:
        df['include'][i] = 0
    for j in range(df['startloc'][i], df['endloc'][i]+1):
        if j in sb2 and tg_label[j] == 'b':
            df['text'][i] = df['text'][i] + ';'
        elif j in sb2 and tg_label[j] == 'sil':
            df['text'][i] = df['text'][i] + ','
        else:
            df['text'][i] = df['text'][i] + tg_text[j]
        if j < df['endloc'][i]:
            df['text'][i] = df['text'][i] + ' '
        else:
            df['text'][i] = df['text'][i] + '.'

#%% create output
df_out = df[df['include']==1]
y = load_wav(orig_wav_root+infiles[epi]+'.wav', sr=sr)
wav_out = np.asarray(y[1])
for i, row in df_out.iterrows():
    wwrite = wav_out[int(row['starttime']*sr):
                     int((row['endtime']+margin)*sr)]
    wwrite = np.concatenate((np.zeros(int(trail*sr)), 
                             wwrite, 
                             np.zeros(int(trail*sr))))
    soundfile.write(output_loc+row['file']+'.wav', wwrite, sr, subtype='PCM_16')
df_out.to_csv(output_root + episode + '_summary.csv')
df.to_csv(output_root + episode + 'full_summary.csv')
