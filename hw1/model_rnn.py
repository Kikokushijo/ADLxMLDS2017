from read_data import Dataset, SentenceBatch
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM

if __name__ == "__main__":
	DS = Dataset()
    SB = SentenceBatch(DS.set_train_label(), batch_size=32)
    batch_gen = SB.collect_sentence()

    model = Sequential
    model.add(LSTM(32, input_shape=(800, 69)))
    model.add(Dropout(0.5))
    model.add(LSTM(16))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='softmax'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	