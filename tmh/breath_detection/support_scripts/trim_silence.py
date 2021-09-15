#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 07:36:29 2020

@author: szekely
"""

from codes.helpers import load_wav, list_filenames
import librosa
import numpy as np
import soundfile

#%%
in_folder = './mtm/mtm_split/'
out_folder = './mtm/mtm/'
files = list(list_filenames(in_folder, ['.wav'], add_ext=False))
#with open('./trim_list.txt', 'r') as f:        
#    files = list(f)
files = [f.rstrip() for f in files]    
files.sort()
sr = 22050
#files[0] = 'sn0801_sent051'
margin = 0.01 # seconds
marginr = 0.01 # use 0.27 for k s t at end of utterance (before breath)

#%%
dur = np.zeros((len(files),4))
#for i in range(100):
for i in range(len(files)):
    y = load_wav(in_folder + files[i] + '.wav', sr=sr)
    dur[i,0] = len(y[1])/sr
    y_out = librosa.effects.trim(y[1], top_db=18)
    if y_out[1][0] > margin*sr:
        start = int( y_out[1][0] - margin*sr )
    else:
        start = 0
    if y_out[1][1] < y[2] - marginr*sr:
        stop = int( y_out[1][1] + marginr*sr )
    else:
        stop = int( y[2] ) # use this by definition for no end trim
    #stop = int( y[2] ) # use this by definition for no end trim
    soundfile.write(out_folder + files[i] + '.wav', y[1][start:stop], sr, subtype='PCM_16')
    dur[i,1] = start/sr
    dur[i,2] = stop/sr
    dur[i,3] = (stop - start)/sr


#%%
with open('./mtm/mtm_extreme_trimmed.txt', 'w') as f:        
     for item in zip(files, dur):
         f.write("%s %s\n" % item)


#%% copy files over


with open('./copy_list.txt', 'r') as f:        
    files = list(f)
files = [f.rstrip() for f in files]    

from shutil import copyfile
in_folder = 'train_input/wavs/M01_trim/'
out_folder = 'train/wavs/M01/'

for file in files:
    copyfile(in_folder + file, out_folder + file)
    
    