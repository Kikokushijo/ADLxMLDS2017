from read_data import Dataset
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from copy import copy
import numpy as np
import os
import sys

if __name__ == "__main__":

    datapath = '../hw1_dataset'
    DS_m = Dataset(datapath, dataset='mfcc')
    DS_f = Dataset(datapath, dataset='fbank')

    DS_m.set_label_dict()
    DS_f.set_label_dict()

    Xf, Yf = DS_f.get_train_data(repeat=True)
    Xm, Ym = DS_m.get_train_data(repeat=True)

    Xc = np.concatenate((Xf, Xm), axis=2)
    Yc = copy(Yf)

    del Xf
    del Yf
    del Xm
    del Ym
    del DS_m
    del DS_f

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='random_normal'), input_shape=(800, 108)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='random_normal')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='random_normal')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='random_normal')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(39, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    filepath = 'models/big_RNN_model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    train_history = model.fit(Xc, Yc, validation_split=0.0, epochs=400, batch_size=128, callbacks=callbacks_list, verbose=1)