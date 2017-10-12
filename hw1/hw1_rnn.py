from read_data import Dataset
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

if __name__ == "__main__":

    DS = Dataset(input())
    DS.set_label_dict()
    key, sentences = DS.get_test_data()
    print(len(key), sentences.shape)

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(800, 69)))
    model.add(LSTM(800, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(48, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    model.load_weights('models/TimitComplexRnnModelAzure.h5')
    predict = model.predict_classes(sentences)
    print(predict[:10])
    
    with open('hw1_rnn_output.csv', 'w') as f:
        f.write('id,phone_sequence\n')
        for index, row in enumerate(predict):
            frame_width = 7
            start = 0
            record = []
            row = [DS.reduce_dict[i] for i in row]
            while start + frame_width < len(row):
                block = row[start:start+frame_width]
                max_occurred = max(block, key=block.count)
                if block.count(max_occurred) >= 3:
                    record.append(max_occurred)
                start += 1
            
            print('Index: ', index, 'result: ', record)
            reduced_result = ""
            for i in range(len(record)-1):
                if record[i] != record[i+1]:
                    reduced_result += DS.char_dict[record[i]]
            f.write('%s,%s\n' % (key[index], reduced_result[1:]))
            print(reduced_result)
            
            
