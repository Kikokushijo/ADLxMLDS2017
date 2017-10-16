import numpy as np
import os

class Dataset(object):
    def __init__(self, path, dataset='fbank'):
        self.read_phone2num()
        self.read_num2char()
        self.feature_dimdict = {'fbank':69, 'mfcc':39}
        self.feature_dim = self.feature_dimdict[dataset]
        self.train_features = []
        self.train_labels = []
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

    def get_train_data(self):
        if not (self.train_features and self.train_labels):
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
                    sentence.label[int(timeid)-1][self.label_dict[name]] = 1.0
                    sentence.length = int(timeid)

                self.train_features = np.vstack([[sentence.feature] for sentence in train_data])
                self.train_labels = np.vstack([[sentence.label] for sentence in train_data])

        return self.train_features, self.train_labels

    def get_test_data(self):
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

            self.test_features = np.vstack([[sentence.feature] for sentence in test_data])
            self.name = [sentence.name for sentence in test_data]

            return self.name, self.test_features

    def shuffle_train_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.train_features)
        np.random.set_state(rng_state)
        np.random.shuffle(self.train_labels)



class Sentence(object):
    def __init__(self, name, feature_dim, max_length=800, char_variety=39):
        self.name = name
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.feature = np.zeros((max_length, feature_dim))
        self.label = np.zeros((max_length, char_variety))
        self.length = 0