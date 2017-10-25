from read_data import Dataset
from model_rnn import Model
import sys
import os

def output_result(filters, output_filename, predict):
    if filters == 'from_class':
        with open(output_filename, 'w') as f:
            f.write('id,phone_sequence\n')
            for index, row in enumerate(predict):
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
                f.write('%s,%s\n' % (key[index][0], result[1:]))
                print(result)
    if filters == 'from_prob':
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

    if filters == 'repeat':
        rows = []
        with open('output/CRC_mfcc_97.csv', 'w') as f:
            f.write('id,phone_sequence\n')
            for index, sentence in enumerate(predict_p):
                
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

                print('Index: ', index, 'result: ', record)
                result = ""
                for i in range(len(record)-1):
                    if record[i] != record[i+1]:
                        result += DS_mfcc.num2char[record[i]]
                f.write('%s,%s\n' % (name[index][0], result[1:]))
                print(result)


if __name__ == "__main__":

    datapath = sys.argv[1]
    DS_m = Dataset(datapath, dataset='mfcc')
    DS_f = Dataset(datapath, dataset='fbank')
    DS_m.set_label_dict()

    set_model = 3
    model_m = select_model(set_model, 'mfcc')
    model_m.load_weights(os.path.join('models', 'RNN_model_mfcc_complex2.h5'))
    predict_m = model.predict_classes(sentences_mfcc)
    
    model_f = select_model(set_model, 'fbank')
    model_f.load_weights(os.path.join('models', 'RNN_model_fbank_complex2.h5'))
    predict_f = model.predict(sentences_fbank)

    output_filename = 'output/mfcc_bid.csv'
    output_result(filter='from_class', output_filename=output_filename, predict=predict_f)