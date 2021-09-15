#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 09:58:08 2021
Train a breath detector using recordings and corresponding annotated textgrids
as a basis. The code has blocks for different types of training: train a new
model from scratch, optionally finetuning it only segments containing breaths
(to strengthen breath detection), or finetune a pretrained model on a target
set of recordings.
@author: szekely
"""


import numpy as np

from scipy.io import wavfile
from multiprocessing import Pool
from functools import partial
from PIL import Image
import random

import os
import codes
from codes.helpers import *
#from utils.list_filenames import *

import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv1D, LSTM
from keras.layers import TimeDistributed, Bidirectional
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

#%% project settings
tg_root = './Liam2/input/TG/' #location of the TextGrid files
wav_root = './Liam2/input/wavs/' #location of the wav files
output_root = './Liam2/input/train_merged/'
#trained_model = './codes/modelMix4.h5'

sr = 44100

slicepersec = 400
secs = 2
n_fft = sr // 50 # standard setting 960
hop_length = n_fft // 8
n_mels=128
timesteps = 40

# keras parameters
batch_size = 16 
epochs = 20
trainpercent = 0.90
seed = 1570
model_name = 'liam2_model2'

#%% preprocess transcript Textgrid to annotation
def transcript2annot(filename,labels,timesteps=40, other=0, merge_breath=False):
    fps = timesteps / 2

    tg = tgio.openTextgrid(filename + ".TextGrid")
    firstTier = tg.tierDict[tg.tierNameList[0]]
    tg_start = [entry[0] for entry in firstTier.entryList]
    tg_stop = [entry[1] for entry in firstTier.entryList]
    tg_label = [entry[2] for entry in firstTier.entryList]
    num_label = [other]*len(tg_label)
    
    frames = int(fps*tg_stop[-1])
    annot = np.zeros(frames).astype(int)
    #labs = list(sorted(set(tg_label)))
    labscheck = [x for x in list(range(len(labels))) if x != other]
    for i in range(len(tg_label)):
        for j in labscheck:
            if tg_label[i].lower().strip() in labels[j]:
                num_label[i] = j
                break
    if merge_breath: # [n, m] if n preceded of followed by m, merge with m
        for i in range(1,len(num_label)-1):
            if num_label[i] == merge_breath[0] and \
                (num_label[i+1] == merge_breath[1] or num_label[i-1] == merge_breath[1]):
                num_label[i] = merge_breath[1]
    for i in range(0,len(num_label)):
        annot[int((tg_start[i]+1/timesteps)*fps):int((tg_stop[i]+1/timesteps)*fps)] = \
        num_label[i]
    return annot

#%% transcription to training labels
files = sorted(list(list_filenames(wav_root, ['.wav'], add_ext=False)))
labels = ['b', 'sp', 'sil', ['xcld', 'start', 'stop', 'singing']]
other = 1


#%% process textgrids to coloured melspectrograms
annot = np.empty((0, timesteps))
specs = np.empty((0, n_mels, secs*slicepersec, 3))

#%% run the processing for episodes and append results
for i in range(5,25):
    # load textgrid
    #temp, labels = textgrid2annot(tg_root + files[i], labels, timesteps=timesteps)
    temp = transcript2annot(tg_root + files[i], labels, timesteps=timesteps, 
                            other=1, merge_breath=[2, 0])
    # load wav file
    y = load_wav(wav_root + files[i] + '.wav', sr=sr)
    adds = min(len(temp) // timesteps, len(y[1]) // (secs*y[3]))
    # append annotations to the annotation array
    annot = np.append(annot, np.reshape(temp[:timesteps * adds], (adds, timesteps)), axis=0)
    # create melspectrograms
    wav_in = np.reshape(y[1][:(secs*y[3]*adds)], (adds,secs*y[3]))
    pool = Pool()
    ins = [wav_in[r, :] for r in range(adds)] 
    melspecs = pool.map(create_melspec, ins)
    # calculate zero crossing rate
    zrates = pool.map(zcr_rate, ins)
    # colour the melspectrograms based on the zero crossing rate
    col_in = [(melspecs[r], zrates[r])
            for r in range(adds)]
    colspecs = pool.map(colorvec2, col_in)
    x_complete = np.asarray(colspecs).astype(np.float32)
    # append the spectrograms
    specs = np.append(specs, x_complete[:,:,:2*slicepersec,:], axis=0)
    del x_complete

#%% remove rows with noise label 3
a_mask = np.any(np.greater(annot, 2.), axis=1)
annot = annot[~a_mask, :]
specs = specs[~a_mask, :, :, :]

#%% display some coloured melspecs
print('random selection:')
indices = random.sample(range(1,np.shape(specs)[0]),min(5,np.shape(specs)[0]))
for i in indices:
    imout = np.floor(255*specs[i-1,::-1,:,:])
    imout = imout.astype(np.uint8)
    print('image:', i)
    img = Image.fromarray(imout, 'RGB')
    img.show()

#%% save the inputs
if not os.path.exists(output_root + 'annot/'):
    os.makedirs(output_root + 'annot/')
if not os.path.exists(output_root + 'zcrgrams/'):
    os.makedirs(output_root + 'zcrgrams/')
    
np.save((output_root + 'annot/annot.npy'), annot)
#np.save((output_root + 'annot/specs.npy'), specs)

for n in range(np.shape(specs)[0]):
    imout = np.floor(255*specs[n,::-1,:,:])
    imout = imout.astype(np.uint8)
    img = Image.fromarray(imout, 'RGB')
    img.save(output_root + 'zcrgrams/zcr' + '{:05d}'.format(n) + '.png','PNG')

#%% load the model input (if necessary)
try:
    annot
except NameError:    
    annot = np.load(output_root + 'annot/annot.npy')

files2 = sorted(list(codes.helpers.list_filenames(output_root + 'zcrgrams', ['.png'])))
im = cv2.imread(files2[0])
x_complete = np.empty((np.shape(files2)[0], np.shape(im)[0], np.shape(im)[1], np.shape(im)[2]))
for j in range(np.shape(files2)[0]):  
    x_complete[j,:,:,:] = cv2.imread(files2[j])
print(np.shape(x_complete))

#%% slice the inputs
random.seed(seed)
index = list(range(np.shape(x_complete)[0]))
random.shuffle(index)
train = int(np.shape(x_complete)[0]*trainpercent)
x_train = x_complete[index[:train],:,:,:]
x_test = x_complete[index[train:],:,:,:]
y_complete = to_categorical(annot)
y_train = y_complete[index[:train],:,:]
y_test = y_complete[index[train:],:,:]
num_classes = np.shape(y_train)[2] #3

#slice the training data into timesteps
img_rows = np.shape(x_train)[1]
img_cols = np.shape(x_train)[2] // timesteps
x2_train = np.empty((np.shape(x_train)[0], timesteps, np.shape(x_train)[1], img_cols, np.shape(x_train)[3]))
for j in range(0,np.shape(x2_train)[0]):
    for k in range(0,timesteps):
        x2_train[j,k,:,:,:] = x_train[j,:,k*img_cols:(k+1)*img_cols,:]

x2_test = np.empty((np.shape(x_test)[0], timesteps, np.shape(x_test)[1], img_cols, np.shape(x_test)[3]))
for j in range(0,np.shape(x2_test)[0]):
    for k in range(0,timesteps):
        x2_test[j,k,:,:,:] = x_test[j,:,k*img_cols:(k+1)*img_cols,:]

#%% save inputs
np.save(output_root + 'x_train_L2.npy',x2_train)
np.save(output_root + 'x_test_L2.npy',x2_test)
np.save(output_root + 'y_train_L2.npy',y_train)
np.save(output_root + 'y_test_L2.npy',y_test)

#%% load all inputs
x2_train = np.load(output_root + 'x_train_L2.npy')
x2_test = np.load(output_root + 'x_test_L2.npy')
y_train = np.load(output_root + 'y_train_L2.npy')
y_test = np.load(output_root + 'y_test_L2.npy')

img_rows = np.shape(x2_train)[2]
img_cols = np.shape(x2_train)[3] #// timesteps
num_classes = np.shape(y_train)[2]

#%% define model
modelZM1 = Sequential()
modelZM1.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3), 
                 strides=(1, 1),
                 activation='relu',
                 padding='same'), input_shape=(timesteps,img_rows,img_cols,3)))
modelZM1.add(TimeDistributed(BatchNormalization()))
modelZM1.add(TimeDistributed(MaxPooling2D(pool_size=(5, 4))))
modelZM1.add(TimeDistributed(Conv2D(8, kernel_size=(4, 1),
                 strides=(4,1),
                 activation='relu')))
modelZM1.add(TimeDistributed(BatchNormalization()))
modelZM1.add(TimeDistributed(MaxPooling2D(pool_size=(6, 5))))
modelZM1.add(TimeDistributed(Flatten()))
modelZM1.add(Bidirectional(LSTM(8, return_sequences=True)))
modelZM1.add(TimeDistributed(Dense(num_classes, activation='softmax')))
modelZM1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
modelZM1.summary()

#%% train model
modelZM1.fit(x2_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x2_test, y_test),
          shuffle=True)

#%% evaluate and save model
score = modelZM1.evaluate(x2_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_prob = modelZM1.predict(x2_test, batch_size=1)
print('prediction shape', np.shape(y_prob))
y_pred = np.argmax(y_prob, axis=2).flatten()
y_act = np.argmax(y_test, axis=2).flatten()

conf = confusion_matrix(y_act, y_pred)

modelZM1.save(output_root + model_name + '.h5')

#%% optional: continue training on dataset containing b in every segment
train_ind = np.where(np.sum(y_train[:,:,0],axis=1)+np.sum(y_train[:,:,2],axis=1)>0)[0]
test_ind = np.where(np.sum(y_test[:,:,0],axis=1)+np.sum(y_test[:,:,2],axis=1)>0)[0]
x3_train = x2_train[train_ind,:,:,:,:]
y3_train = y_train[train_ind,:,:,]
x3_test = x2_test[test_ind,:,:,:,:]
y3_test = y_test[test_ind,:,:,]

#%% number of epochs for continued training
epoch2 = 5

#%% continue training
modelZM1.fit(x3_train, y3_train,
          batch_size=batch_size,
          epochs=epoch2,
          verbose=1,
          validation_data=(x3_test, y3_test))

#%% evaluate and save model
#in sample test
score_v2 = modelZM1.evaluate(x3_test, y3_test, verbose=0)
print('Test loss:', score_v2[0])
print('Test accuracy:', score_v2[1])

y_prob_v2 = modelZM1.predict(x2_test, batch_size=1)
print('prediction shape', np.shape(y_prob_v2))
y_pred_v2 = np.argmax(y_prob_v2, axis=2).flatten()
y_act_v2 = np.argmax(y_test, axis=2).flatten()

conf2 = confusion_matrix(y_act_v2, y_pred_v2)

#total test set evaluation
score_v3 = modelZM1.evaluate(x2_test, y_test, verbose=0)
print('Test loss:', score_v3[0])
print('Test accuracy:', score_v3[1])

y_prob_v3 = modelZM1.predict(x2_test, batch_size=1)
print('prediction shape', np.shape(y_prob_v3))
y_pred_v3 = np.argmax(y_prob_v3, axis=2).flatten()

confusion_matrix(y_act, y_pred_v3)

modelZM1.save(output_root + model_name + '_v2.h5')

#%% transfer learn
# swap column 1 and 2 to align to pre-trained model
y4_train = y_train[:,:,[0,2,1]]
y4_test = y_test[:,:,[0,2,1]]
modelNT = keras.models.load_model(trained_model, compile=False)
modelNT.summary()

#%% continue training
modelNT.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
modelNT.fit(x2_train, y4_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x2_test, y4_test))

#%% evaluate and save model
score_v4 = modelNT.evaluate(x2_test, y4_test, verbose=0)
print('Test loss:', score_v4[0])
print('Test accuracy:', score_v4[1])

y_prob_v4 = modelNT.predict(x2_test, batch_size=1)
print('prediction shape', np.shape(y_prob))
y_pred_v4 = np.argmax(y_prob_v4, axis=2).flatten()
y_act_v4 = np.argmax(y4_test, axis=2).flatten()


confusion_matrix(y_act_v4, y_pred_v4)

modelNT.save(output_root + model_name + '_NT.h5')

