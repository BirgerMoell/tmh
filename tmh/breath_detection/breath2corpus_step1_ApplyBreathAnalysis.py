#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:09:45 2018
Processing of corpus to breath separated utterances using a 
model trained on earlier annotated data.
@author: szekely
"""

#%%   load dependencies
import os
import numpy as np
import csv

from multiprocessing import Pool
from functools import partial

import codes
from codes.helpers import load_wav, create_melspec, zcr_rate, colorvec, colorvec2
from codes.helpers import list_filenames, annot2textgrid
#from codes.helpers import annot2textgrid

from scipy import signal
from scipy import io
from scipy.io import wavfile

import soundfile

from PIL import Image
#import random

import keras
import cv2
 
#%%    settings for the input file

# settings
data_root = './Cormac/output/' # location to save wav and textgrid
input_root = './Cormac/denoised_44k/' # location of wav input files
infiles = list(list_filenames(input_root, extensions='.wav', add_ext=False))
infiles.sort()
output_root = './Cormac/output/'  # location of other outputs from the process
epi = -1 # initialise manual loop

#%% execute process one episode at a time
epi = epi+1
input_file = infiles[epi] + '.wav'

trained_model = './codes/liam2_model_v2.h5' # pretrained model forbreath identification
nr_speakers = 1 # pretrained model is single speaker: setting is always 1
output_prefix = 'C'+infiles[epi] # naming convention for output files
output_start = 0 # if processing multiple wavs in same output, add to the startnumber
create_wav = False # whether to create output samples (not required)

# output settings, don't change these but make sure input files are 44.1k mono
sr = 44100 # desired frequency of output
n_mels = 128
timesteps = 40 # timesteps in keras model, standard 2 seconds with 20 steps each
slicepersec = 400

# corpus creation settings
max_utt = 12 # maximum utterance length (seconds)
min_utt = 1.0 # minimum utterance length (seconds)
trail = 0.02 # silence before and after utterance (seconds)
trailfr = 0 # additional frame(s) included for smoother end of utterance (1 for TCC / 0 for Obama)

# derived settings
n_fft = sr // 50 # standard setting 960
hop_length = n_fft // 8

if not os.path.exists(output_root + output_prefix):
    os.makedirs(output_root + output_prefix)

# load input wav and split into two second samples
y = load_wav(input_root+input_file, sr=sr)
wav_out = np.asarray(y[1])
samples = len(wav_out) // (2 * sr)
wav_in = np.reshape(wav_out[:(2*sr*samples)], (samples,2*sr))

# create mel-spectrogram
pool = Pool()
ins = [wav_in[r, :] 
        for r in range(samples)] 

melspecs = pool.map(create_melspec, ins)

# use zero crossing rate to create zcr-colored melspectrograms
zrates = pool.map(zcr_rate, ins)
func = partial(colorvec, melspecs, zrates)
col_in = [(melspecs[r], zrates[r])
            for r in range(samples)]
colspecs = pool.map(colorvec2, col_in)
x_complete = np.asarray(colspecs).astype(np.float32)
print(np.shape(x_complete))

# save the zcr-coloured melspectrograms
if not os.path.exists(output_root + 'zcrgrams/' + output_prefix):
    os.makedirs(output_root + 'zcrgrams/' + output_prefix)
for n in range(0, np.shape(x_complete)[0]):
    imout = np.floor(255*x_complete[n,::-1,:,:])
    imout = imout.astype(np.uint8)
    img = Image.fromarray(imout, 'RGB')
    img.save(output_root + 'zcrgrams/' + output_prefix + '/zcr' + '{:04d}'.format(n) + '.png','PNG')

soundfile.write(data_root + output_prefix + '.wav', y[1], y[3], subtype='PCM_16')
    
# clean up
del imout, ins, wav_in, y, x_complete, zrates, melspecs, colspecs, col_in
#% data preparation
# prepare the coloured spectrograms to take them through the model
files = list(list_filenames(output_root + 'zcrgrams/' + output_prefix, ['.png']))
files.sort()
print('number of files:', len(files))

im = cv2.imread(files[0])
# x_complete is recreated from the saved spectrograms (saving and loading it
# changed the output)
x_complete = np.empty((np.shape(files)[0], np.shape(im)[0], np.shape(im)[1], np.shape(im)[2]))
for j in range(0,np.shape(files)[0]):  
    x_complete[j,:,:,:] = cv2.imread(files[j])
    
# input image dimensions
img_rows, img_cols = np.shape(x_complete)[1], np.shape(x_complete)[2]

#take the spectogram apart into slices to be fed to the model
img_cols2 = img_cols // timesteps
print('frames per model input slice:', img_cols2)

x2_pred = np.empty((np.shape(x_complete)[0], timesteps, np.shape(x_complete)[1], img_cols2, np.shape(x_complete)[3]))

for j in range(0,np.shape(x_complete)[0]):
    for k in range(0,timesteps):
        x2_pred[j,k,:,:,:] = x_complete[j,:,k*img_cols2:(k+1)*img_cols2,:]
print('model input shape:', np.shape(x2_pred))

# make predictions
model = keras.models.load_model(trained_model, compile=False)
out_pred = model.predict(x2_pred, batch_size=1)
print('prediction shape', np.shape(out_pred))

# merge prediction into one time series
flat_pred = np.empty((np.shape(out_pred)[0]*np.shape(out_pred)[1], np.shape(out_pred)[2]))
for j in range(0,np.shape(out_pred)[0]):
    flat_pred[j*timesteps:(j+1)*timesteps,:] = out_pred[j,:,:]
print('flattened prediction shape:', np.shape(flat_pred))


# clean up and save results
if not os.path.exists(output_root + 'predictions/'):
    os.makedirs(output_root + 'predictions/')
np.save(output_root + 'predictions/' + output_prefix + '.npy', flat_pred)
del out_pred, x2_pred, x_complete
#% process output
# this code is a simplified version, providing no speaker separation 
# the pretrained model is aimed at separating breath groups
#    two speaker labels: {0: noise/music, 1: breath s1, 2: breath s2, 3: silence,
#    4: speech s1, 5: speech s2, 6: segment with mixed speech}
#    NT corpus single speaker original labels: {0: breath, 1: pause, 2: speech}
#    LV corpus original labels: {0: breath, 1: speech, 2: noise}

try:
    flat_pred
except NameError:
    flat_pred = np.load(output_root + 'predictions/' + output_prefix + '.npy')
#flat_pred = np.load(output_root + 'predictions/ep' + episode + '.npy')

#    prepare output
pred = np.argmax(flat_pred, axis=1)
# merge segments of only 1 slice into previous segment
for j in range(1,len(pred)-1): 
    if (pred[j] != pred[j-1] and pred[j] != pred[j+1]):
        pred[j] = pred[j-1]

# align single speaker labelling to two speaker case
if nr_speakers == 1:
    pred[pred == 1] = 4 # speech
    pred[pred == 0] = 1 # breath
    pred[pred == 2] = 3 # silence

w_change = np.where(pred[:-1] != pred[1:])[0]+1
w_id = pred[w_change] # identify changes in segment (breath, speech, speaker)
w_br = np.where(np.logical_or(w_id == 1, w_id == 2))
w_br = np.where((w_id == 1) | (w_id == 2))[0] # identify breaths 2 speakers

# create annotation textgrid
filename = data_root + output_prefix
labels = ["n", "b", "na", "sil", "sp"]

annot2textgrid(filename,labels,pred,timesteps=40)

#% create utterance specification and output summary
fps = int(timesteps/2)

# create output summary
output = np.zeros((len(w_br)-1,21))
for i in range(0,len(w_br)-1):
    if output[i,0] == -1:
        output[i,0] = 0
    else:
        output[i,0] = w_id[w_br[i]] # breath id
        output[i,20] = w_br[i] + 2 # likely speech id
        output[i,1] = w_change[w_br[i]] # start frame
        try:
            output[i,2] = w_change[w_br[i+1] + 1] # stop frame
        except IndexError:
            output[i,2] = len(w_change)
            output[i,0] = 0
        output[i,3] = w_change[w_br[i]+1]-w_change[w_br[i]] # start breath in frames
        output[i,4] = w_change[w_br[i+1]]-w_change[w_br[i]+1] # frames from start of speech to breath
        k = i
        while output[i,4] < min_utt*fps and k < len(w_br)-3:   # merge with next breathgroup if too short
            k = k+1
            output[i,2] = w_change[w_br[k+1] + 1] # stop frame
            output[i,4] = w_change[w_br[k+1]]-w_change[w_br[i]+1] # frames from start of speech to breath
            output[k,0] = -1 # skip the next utterance
        if output[i,4] > max_utt*fps:   # change end frame to last silence if longer than maximum length
            ends = np.where(pred[w_change[w_br[i]+1]:w_change[w_br[i]+1]+max_utt*fps]==3)[0]
            if len(ends) > 0:
                output[i,2] = w_change[w_br[i]+1] + ends[-1]
                output[i,5] = output[i,4] # speech time before shortening
                output[i,4] = ends[-1]
                if len(np.where(ends[:-1] != (ends[1:]-1))[0]) > 0:
                    output[i,6] = len(ends) - np.where(ends[:-1] != (ends[1:]-1))[0][-1] - 1 # closing silence in frames
                else:
                    output[i,6] = len(ends)
            else:
                output[i,0] = 0
                output[i,5] = output[i,4] # speech time of too short utterance
                output[i,6] = output[i,2]-output[i,1]-output[i,4]-output[i,3] # closing breath in frames        
        else:
            output[i,6] = output[i,2]-output[i,1]-output[i,4]-output[i,3] # closing breath in frames        
        output[i,7] = output[i,1]/fps # start time
        output[i,8] = (output[i,2]+trailfr)/fps # stop time
        output[i,9] = output[i,8]-output[i,7] # duration incl breaths
        for j in range(7):
            output[i,11+j] = np.count_nonzero(pred[int(output[i,1]):int(output[i,2])]==j)

output = output[output[:,0] == 1,:]

if not os.path.exists(output_root + output_prefix):
    os.makedirs(output_root + output_prefix) 

# create annotation textgrid (Praat) to improve annotation and train additional models
#filename = data_root + input_file[:-4]
#labels = ["na0", "b", "na2", "sil", "sp"]
#
#annot2textgrid(filename,labels,pred,timesteps=40)
        
#% create wav files
if not os.path.exists(output_root + output_prefix + '/wav'):
    os.makedirs(output_root + output_prefix + '/wav') 

try:
    wav_out
except NameError:
    y = load_wav(data_root+input_file, sr=sr)
    wav_out = np.asarray(y[1])
        
wavname = [None] * len(output)
for i in range(len(output)):
    wwrite = wav_out[int(output[i,7]*sr):int(output[i,8]*sr)]
    output[i,18] = np.where(np.abs(wwrite) < 1e-5)[0][0]
    output[i,19] = np.where(np.abs(wwrite) < 1e-5)[0][-1]
    wwrite = np.concatenate((np.zeros(int(trail*sr)), 
                            wwrite[np.where(np.abs(wwrite) < 1e-5)[0][0]:
                            np.where(np.abs(wwrite) < 1e-5)[0][-1]], 
                            np.zeros(int(trail*sr))))
    wavname[i] = (output_root + output_prefix + '/wav/'+ 
              output_prefix + '_' + '{:04d}'.format(i+output_start) + '.wav')
    #wavfile.write(wavname[i], sr, wwrite)
    if create_wav:
        soundfile.write(wavname[i], wwrite, sr, subtype='PCM_16')
        
#% save output statistics
np.save(output_root + output_prefix + '/output_' + output_prefix + '.npy', output)
np.savetxt(output_root + output_prefix + '/output_' + output_prefix + '.csv', output, delimiter=",") 
