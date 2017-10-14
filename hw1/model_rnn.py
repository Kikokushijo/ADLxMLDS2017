from read_data38 import Dataset
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
import os

def select_model(set_model):
    if set_model == 0:
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(800, 69)))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(TimeDistributed(Dense(39, activation='softmax')))
        optimizer = optimizers.RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
    if set_model == 1:
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(800, 69)))
        model.add(LSTM(800, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(TimeDistributed(Dense(128, activation='relu')))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(39, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        print(model.summary())
    if set_model == 2:
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(800, 69)))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(200, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(128, activation='relu')))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(39, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        print(model.summary())
    return model

if __name__ == "__main__":
    DS = Dataset(input())
    # DS = Dataset(r'../hw1_dataset')
    DS.set_label_dict()
    X, Y = DS.get_train_data()
    print(X.shape, Y.shape)

    set_model = 2
    model = select_model(set_model)

    for _ in range(10):
        try:
            model.load_weights(os.path.join('models', 'RNN_model%d.h5' % (set_model)))
        except:
            pass

        DS.shuffle_train_data()
        train_history = model.fit(X, Y, batch_size=128, epochs=1, verbose=1, validation_split=0.2)
        model.save_weights(os.path.join('models', 'RNN_model%d.h5' % (set_model)))