from read_data38 import Dataset
from model_rnn import select_model
import os

if __name__ == "__main__":

    DS = Dataset(input(), dataset='mfcc')
    output_filename = input()
    key, sentences = DS.get_test_data()
    print(len(key), sentences.shape)

    set_model = 2
    model = select_model(set_model)
    model.load_weights(os.path.join('models', 'RNN_model_mfcc%d.h5' % (set_model)))
    predict = model.predict_classes(sentences)
    
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
            
            
