import numpy as np
import os

class Dataset(object):
    def __init__(self, path, dataset='fbank'):
        self.read_phone2num()
        self.read_num2char()
        self.feature_dimdict = {'fbank':69, 'mfcc':39}
        self.feature_dim = self.feature_dimdict[dataset]
        self.train_features = []
        self.train_labels_onehot = []
        self.train_labels = []
        self.test_features = []
        self.name = []
        self.train_data_path = os.path.join(path, dataset, 'train.ark')
        self.test_data_path = os.path.join(path, dataset, 'test.ark')
        self.label_dict_path = os.path.join(path, 'label', 'train.lab')

    def read_phone2num(self):
        self.phone2num = {}
        with open('dict/phone2num') as f:
            for line in f:
                phone, num = line.strip('\n').split(' ')
                self.phone2num[phone] = int(num)

    def read_num2char(self):
        self.num2char = {}
        with open('dict/num2char') as f:
            for line in f:
                num, char = line.strip('\n').split(' ')
                self.num2char[int(num)] = char

    def set_label_dict(self):
        self.label_dict = {}
        with open(self.label_dict_path, 'r') as f:
            for line in f:
                name, phone = line.strip('\n').split(',')
                self.label_dict[name] = self.phone2num[phone]

    def get_train_data(self, padding=False):
        if len(self.train_features) == 0  and len(self.train_labels_onehot) == 0:
            with open(self.train_data_path, 'r') as f:
                train_data = []
                sentence = None
                for line in f:
                    name, *features = line.strip('\n').split(' ')
                    spkird, sentid, timeid = name.split('_')
                    key = "%s_%s" % (spkird, sentid)

                    if len(train_data) == 0 or train_data[-1].name != key:
                        sentence = Sentence(key, self.feature_dim)
                        train_data.append(sentence)
                    sentence.feature[int(timeid)-1][:] = np.array(features).astype(np.float)
                    sentence.label_onehot[int(timeid)-1][self.label_dict[name]] = 1.0
                    sentence.label[int(timeid)-1] = self.label_dict[name]
                    sentence.length = int(timeid)


                if padding:
                    for sentence in train_data:
                        length = sentence.length
                        max_length = sentence.max_length
                        padding_times = max_length // length
                        if padding_times > 1:
                            for i in range(1, padding_times):
                                sentence.feature[length*i:length*(i+1)] = sentence.feature[:length]
                                sentence.label_onehot[length*i:length*(i+1)] = sentence.label_onehot[:length]
                                sentence.label[length*i:length*(i+1)] = sentence.label[:length]
                                # for index, row in enumerate(sentence.label[:length]):
                                #     print('index: ', index, 'row: ', row)
                        sentence.feature[length*padding_times:max_length] = sentence.feature[length-1]
                        sentence.label_onehot[length*padding_times:max_length] = sentence.label_onehot[length-1]
                        sentence.label[length*padding_times:max_length] = sentence.label[length-1]

                self.train_features = np.vstack([[sentence.feature] for sentence in train_data])
                self.train_labels_onehot = np.vstack([[sentence.label_onehot] for sentence in train_data])
                self.train_labels = np.vstack([[sentence.label] for sentence in train_data])

        return self.train_features, self.train_labels_onehot

    def get_test_data(self, padding=False):
        if len(self.test_features) > 0 and len(self.name) > 0:
            with open(self.test_data_path, 'r') as f:
                test_data = []
                sentence = None
                for line in f:
                    name, *features = line.strip('\n').split(' ')
                    spkird, sentid, timeid = name.split('_')
                    key = "%s_%s" % (spkird, sentid)

                    if len(test_data) == 0 or test_data[-1].name != key:
                        sentence = Sentence(key, self.feature_dim)
                        test_data.append(sentence)
                    sentence.feature[int(timeid)-1][:] = np.array(features).astype(np.float)
                    sentence.length = int(timeid)

                if padding:
                    for sentence in test_data:
                        length = sentence.length
                        max_length = sentence.max_length
                        padding_times = max_length // length
                        if padding_times > 1:
                            for i in range(1, padding_times):
                                sentence.feature[length*i:length*(i+1)] = sentence.feature[:length]

                        sentence.feature[length*padding_times:max_length] = sentence.feature[length-1]

                self.test_features = np.vstack([[sentence.feature] for sentence in test_data])
                self.name = [(sentence.name, sentence.length) for sentence in test_data]

        return self.name, self.test_features

    def shuffle_train_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.train_features)
        np.random.set_state(rng_state)
        np.random.shuffle(self.train_labels_onehot)



class Sentence(object):
    def __init__(self, name, feature_dim, max_length=800, char_variety=39):
        self.name = name
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.feature = np.zeros((max_length, feature_dim))
        self.label_onehot = np.zeros((max_length, char_variety))
        self.label = np.zeros(max_length)
        self.length = 0