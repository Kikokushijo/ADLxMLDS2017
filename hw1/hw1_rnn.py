from read_data import Dataset
from model_rnn import select_model
import os

def output_result(filter, output_filename, predict):
    if filter == 'from_class':
        with open(output_filename, 'w') as f:
            f.write('id,phone_sequence\n')
            for index, row in enumerate(predict):
                frame_width = 7
                start = 0
                record = []
                while start + frame_width < len(row):
                    block = row[start:start+frame_width]
                    max_occurred = max(block, key=block.count)
                    if block.count(max_occurred) >= 4:
                        record.append(max_occurred)
                    start += 1
                
                print('Index: ', index, 'result: ', record)
                result = ""
                for i in range(len(record)-1):
                    if record[i] != record[i+1]:
                        result += DS.num2char[record[i]]
                f.write('%s,%s\n' % (key[index], result[1:]))
                print(result)
    if filter == 'from_prob':
        with open('mfcc_fbank.csv', 'w') as f:
            f.write('id,phone_sequence\n')
            for index, sentence in enumerate(mfcc_fbank):
                row = []
                for vector in sentence:
                    classes, prob = max(enumerate(vector), key=lambda x:x[1])
                    row.append(classes)
                
                frame_width = 7
                start = 0
                record = []
                while start + frame_width < len(row):
                    block = list(row[start:start+frame_width])
                    max_occurred = max(block, key=block.count)
                    if block.count(max_occurred) >= 4:
                        record.append(max_occurred)
                    start += 1

                print('Index: ', index, 'result: ', record)
                result = ""
                for i in range(len(record)-1):
                    if record[i] != record[i+1]:
                        result += DS.num2char[record[i]]
                f.write('%s,%s\n' % (key[index], result[1:]))
                print(result)

if __name__ == "__main__":

    datapath = input()
    DS_mfcc = Dataset(datapath, dataset='mfcc')
    # output_filename = input()
    mfcc_output_filename = 'mfcc_5.csv'
    key_mfcc, sentences_mfcc = DS_mfcc.get_test_data()
    print(len(key_mfcc), sentences_mfcc.shape)

    DS_fbank = Dataset(datapath, dataset='fbank')
    fbank_output_filename = 'fbank_5.csv'
    key_fbank, sentences_fbank = DS_fbank.get_test_data()
    print(len(key_fbank), sentences_fbank.shape)

    set_model = 3
    model_m = select_model(set_model, 'mfcc')
    model_m.load_weights(os.path.join('models', 'RNN_model_mfcc_complex2.h5'))
    predict_m = model.predict_classes(sentences_mfcc)
    
    model_f = select_model(set_model, 'fbank')
    model_f.load_weights(os.path.join('models', 'RNN_model_fbank_complex2.h5'))
    predict_f = model.predict(sentences_fbank)

    output_filename = 'output/mfcc_bid.csv'
    output_result(filter='from_class', output_filename=output_filename, predict=predict_f)