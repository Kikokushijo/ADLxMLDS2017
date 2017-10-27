from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys
import os

from read_data import Dataset

def output_result(predict, name):
    rows = []
    with open(sys.argv[2], 'w') as f:
        f.write('id,phone_sequence\n')
        for index, sentence in enumerate(predict):

            length = name[index][1]
            padding_times = 800 // length
            prob_sequence = []
            for v_index, v_i in enumerate(sentence[:length]):
                for _ in range(1, padding_times):
                    v_i += sentence[v_index + _ * length]
                prob_sequence.append(v_i)

            assert len(prob_sequence) == length

            row = []
            for vector in prob_sequence:
                classes, prob = max(enumerate(vector), key=lambda x:x[1])
                row.append(classes)
            rows.append(row)

            assert len(row) == length

            frame_width = 7
            start = 0
            record = []
            while start + frame_width < len(row):
                block = list(row[start:start+frame_width])
                max_occurred = max(block, key=block.count)
                if block.count(max_occurred) >= 3:
                    record.append(max_occurred)
                start += 1
            result = ""
            for i in range(len(record)-1):
                if record[i] != record[i+1]:
                    result += DS_m.num2char[record[i]]
            f.write('%s,%s\n' % (name[index][0], result[1:]))

def set_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=11, padding='same', activation='relu', input_shape=(800, 108), kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=11, padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(64, activation='relu', kernel_initializer='he_normal')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(39, activation='softmax', kernel_initializer='he_normal')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == "__main__":

    datapath = sys.argv[1]
    DS_m = Dataset(datapath, dataset='mfcc')
    DS_f = Dataset(datapath, dataset='fbank')
    DS_m.set_label_dict()

    key, Xft = DS_f.get_test_data(repeat=True)
    key, Xmt = DS_m.get_test_data(repeat=True)
    Xct = np.concatenate((Xft, Xmt), axis=2)

    model = set_model()
    filepath = "models/CRNN_model.h5"
    model.load_weights(filepath)
    predict = model.predict(Xct)

    output_result(predict, key)