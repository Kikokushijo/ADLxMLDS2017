from read_data import Dataset
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM


if __name__ == "__main__":
    DS = Dataset(r'D:/Data/hw1_dataset')
    DS.set_label_dict()
    X, Y = DS.get_train_data()
    print(X.shape, Y.shape)

    # model = Sequential()
    # model.add(Masking(mask_value=0.0, input_shape=(800, 69)))
    # model.add(LSTM(100, return_sequences=True))
    # model.add(Dropout(0.4))
    # model.add(TimeDistributed(Dense(48, activation='softmax')))
    # optimizer = optimizers.RMSprop(lr=0.01)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # print(model.summary())

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(800, 69)))
    model.add(LSTM(800, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(48, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(model.summary())

    for _ in range(10):
        try:
            model.load_weights('D:/Data/ADLxMLDS2017/hw1/models/TimitComplexRnnModel.h5')
        except:
            pass

        train_history = model.fit(X, Y, batch_size=32, epochs=1, verbose=1, validation_split=0.2)
        model.save_weights('D:/Data/ADLxMLDS2017/hw1/models/TimitComplexRnnModel.h5')