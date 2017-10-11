from read_data import Dataset
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

if __name__ == "__main__":
    DS = Dataset(r'C:/Users/User/Desktop/dataset')
    X, Y = DS.set_train_label()

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(800, 69)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(48, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(model.summary())

    for _ in range(10):
        try:
            model.load_weights('D:/Data/ADLxMLDS2017/hw1/models/TimitRnnModel.h5')
        except:
            pass

        train_history = model.fit(X, Y, batch_size=100, epochs=10, verbose=1, validation_split=0.2)
        model.save_weights('D:/Data/ADLxMLDS2017/hw1/models/TimitRnnModel.h5')