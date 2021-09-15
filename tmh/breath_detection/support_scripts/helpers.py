#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:50:09 2018
helper functions for stp_episode 
@author: szekely
interpreter: std spyder
"""

# 
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from scipy import signal
from scipy import io
from scipy.io import wavfile
import soundfile as sf

from praatio import tgio

import librosa
import librosa.display

import pandas as pd


#%% load_wav function
def load_wav(fn, sr=None, normalize=True):
    if fn == '': # ignore empty filenames
        print('filename missing')
        return None
    fs, audio = wavfile.read(fn)
    audio = audio.astype(np.float32)
    duration = np.shape(audio)[0]
    if duration == 0: # ignore zero-length samples
        print('sample has no length')
        return None
    if sr != fs and sr != None:
        audio = librosa.resample(audio, fs, sr)
        fs = sr
    max_val = np.abs(audio).max()
    if max_val == 0: # ignore completely silent sounds
        print('silent sample')
        return None
    if normalize:
        audio = audio / max_val
    #audio = audio.astype(np.int16)
    return (fn, audio, duration, fs)

#%% create_melspec function
def create_melspec(wav_in, sr=None, n_fft = 960, hop_length=120, n_mels=128):
    if sr == None:
        sr = min(48000, len(wav_in) // 2)
        n_fft = sr // 50
        hop_length = sr // 400
    S = librosa.feature.melspectrogram(wav_in, power=1, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    #log_S = librosa.amplitude_to_db(S, ref=np.max)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    melspecs = np.asarray(log_S).astype(np.float32)
    return melspecs

def load_wav_16bit(fn, sr=None, normalize=True):
    if fn == '': # ignore empty filenames
        print('filename missing')
        return None
    audio, fs = sf.read(fn,dtype='int16')
    #audio = audio.astype(np.float32)
    duration = np.shape(audio)[0]
    if duration == 0: # ignore zero-length samples
        print('sample has no length')
        return None
    return (fn, audio, duration, fs)

#%%
def normalise(y):
    if np.amax(y) == np.amin(y):
        print('ERROR: max and min values are equal')
        return None
    y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
    return y

#%% zcr function
def zcr_rate(wav_in, step=240, sz=960):
    #if len(wav_in) < 2*48000:
    #    sz = len(wav_in) // (2*50)
    #    step = len(wav_in) // (2*200)
    cross = np.abs(np.diff(np.sign(wav_in+1e-8)))
    cross[cross > 1] = 1
    steps=int((np.shape(cross)[0] - sz) / step)
    zrate=np.zeros((steps,))
    for i in range(steps):
        zrate[i]=np.mean(cross[i*step:i*step+sz])
    return zrate

#%% list filenames in a folder
def list_filenames(directory, extensions=None, add_ext=True):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                if add_ext:
                    yield joined
                else:
                    yield base

def list_filelocations(directory, extensions=None, add_ext=True):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                if add_ext:
                    yield joined
                else:
                    yield base, root


#%% vectorised implementation
def colorvec(spec, zcrate, maxzcr=0.4, low_slow=np.array([0., 255., 255.]), 
              low_fast=np.array([255., 255., 255.]), 
              high_slow=np.array([0., 0., 0.]), 
              high_fast=np.array([255., 0., 0.])):
    zcr2 = np.interp(range(np.shape(spec)[1]), np.linspace(0, np.shape(spec)[1], np.shape(zcrate)[0]), zcrate )
    spec2 = np.abs(spec)/80
    outp = np.zeros((np.shape(spec2)[0], np.shape(spec2)[1], 3))
    z = zcr2/maxzcr
    z[z > 1] = 1
    low = np.outer(low_slow,np.ones_like(z)) + np.outer(low_fast - low_slow,z)
    high = np.outer(high_slow,np.ones_like(z)) + np.outer(high_fast - high_slow,z)
    for k in range(3):
        outp[:,:,k] = np.tile(low[k,:], (np.shape(spec2)[0], 1)) + spec2 * np.tile(high[k,:] - low[k,:], (np.shape(spec2)[0], 1))
    outp = outp / 255
    return outp

#%% vectorised implementation
def colorvec2(inp, maxzcr=0.4, low_slow=np.array([0., 255., 255.]), 
              low_fast=np.array([255., 255., 255.]), 
              high_slow=np.array([0., 0., 0.]), 
              high_fast=np.array([255., 0., 0.])):
    zcr2 = np.interp(range(np.shape(inp[0])[1]), np.linspace(0, np.shape(inp[0])[1], np.shape(inp[1])[0]), inp[1] )
    spec2 = np.abs(inp[0])/80
    outp = np.zeros((np.shape(spec2)[0], np.shape(spec2)[1], 3))
    z = zcr2/maxzcr
    z[z > 1] = 1
    low = np.outer(low_slow,np.ones_like(z)) + np.outer(low_fast - low_slow,z)
    high = np.outer(high_slow,np.ones_like(z)) + np.outer(high_fast - high_slow,z)
    for k in range(3):
        outp[:,:,k] = np.tile(low[k,:], (np.shape(spec2)[0], 1)) + spec2 * np.tile(high[k,:] - low[k,:], (np.shape(spec2)[0], 1))
    outp = outp / 255
    return outp

#%% preprocess Textgrid to annotation
def textgrid2annot(filename,labels,timesteps=40):
    fps = timesteps / 2

    tg = tgio.openTextgrid(filename + ".TextGrid")
    firstTier = tg.tierDict[tg.tierNameList[0]]
    tg_start = [entry[0] for entry in firstTier.entryList]
    tg_stop = [entry[1] for entry in firstTier.entryList]
    tg_label = [entry[2] for entry in firstTier.entryList]
    
    frames = int(fps*tg_stop[-1])
    annot = np.zeros(frames).astype(int)
    labs = list(sorted(set(tg_label)))
    for i in range(len(labs)):
        if labs[i] not in labels:
            labels.append(labs[i])
    for i in range(0,len(tg_label)):
        annot[int((tg_start[i]+1/timesteps)*fps):int((tg_stop[i]+1/timesteps)*fps)] = \
        labels.index(tg_label[i])
    return annot, labels

def annot2textgrid(filename,labels,annot,timesteps=40):
    fps = timesteps / 2

    annotTier = tgio.IntervalTier('annot', [], 0, pairedWav=filename + '.wav')
    
    tg = tgio.Textgrid()
    tg.addTier(annotTier)
    
    w_change = np.where(annot[:-1] != annot[1:])[0]+1
    w_id = annot[w_change]
    #timegrid = np.zeros((len(w_change),2))
    annotTier.insertEntry((0, w_change[0]/fps, labels[annot[0]]), warnFlag=True, collisionCode='replace')    
    for i in range(len(w_change)-1):
        annotTier.insertEntry((w_change[i]/fps, w_change[i+1]/fps, labels[w_id[i]]), warnFlag=True, collisionCode='replace')
    annotTier.insertEntry((w_change[-1]/fps, len(annot)/fps, labels[annot[-1]]), warnFlag=True, collisionCode='replace')        
    tg.save(filename + ".TextGrid")
    
def annot_txt2textgrid(filename,labels= ['b', 'sp', 'sil'],annot_labels = ['@', '', 'sil'], checktimes=True):

    annotTier = tgio.IntervalTier('annot', [], 0, pairedWav=filename + '.wav')
    textTier = tgio.IntervalTier('text', [], 0, pairedWav=filename + '.wav')
    phonemeTier = tgio.IntervalTier('phon', [], 0, pairedWav=filename + '.wav')
    
    tg = tgio.Textgrid()
    tg.addTier(annotTier)
    tg.addTier(textTier)
    tg.addTier(phonemeTier)

    input = pd.read_csv(filename+'.txt', sep="\t", header=None, encoding='UTF-8')
    
    if checktimes:
        for i in range(len(input)-1):
            if input[1][i+1] < input[2][i]:
                mid = 0.5 * (input[1][i+1] + input[2][i])
                input[1][i+1] = mid
                input[2][i] = mid
    
    for i in range(len(input)):
        if input[4][i] != input[4][i]:
            if (input[3][i].strip() in annot_labels):
                annotTier.insertEntry((input[1][i], input[2][i], 
                                       labels[annot_labels.index(input[3][i].strip())]), 
                                      warnFlag=True, collisionCode='replace')
                textTier.insertEntry((input[1][i], input[2][i], input[3][i]), 
                                      warnFlag=True, collisionCode='replace')
                phonemeTier.insertEntry((input[1][i], input[2][i], ''), 
                                      warnFlag=True, collisionCode='replace')
            else:
                print(f'incorrect entry (missing phonemic transcript or incorrect label) in entry {i}')
        else:
            annotTier.insertEntry((input[1][i], input[2][i], 
                                   labels[annot_labels.index('')]), 
                                  warnFlag=True, collisionCode='replace')
            textTier.insertEntry((input[1][i], input[2][i], input[3][i]), 
                                  warnFlag=True, collisionCode='replace')
            phonemeTier.insertEntry((input[1][i], input[2][i], input[4][i]), 
                                  warnFlag=True, collisionCode='replace')
    tg.save(filename + ".TextGrid")
    
    