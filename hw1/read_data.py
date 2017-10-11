import numpy as np
import os

class Dataset(object):
    def __init__(self, path='../hw1_dataset/'):

        self.char_dict = {}
        self.label_dict = {}
        self.reduce_dict = {}
        
        self.char_dict_path = os.path.join(path, '48phone_char.map')
        self.set_char_dict()

        self.label_dict_path = os.path.join(path, 'label', 'train.lab')
        self.set_label_dict()

        self.reduce_dict_path = os.path.join(path, 'phones', '48_39.map')
        self.set_reduce_dict()

        self.train_set_path = os.path.join(path, 'fbank', 'train.ark')

    def set_char_dict(self):
        with open(self.char_dict_path) as f:
            for line in f.readlines():
                phone, num, char = line.strip('\n').split('\t')
                self.char_dict[phone] = [int(num), char]

    def set_label_dict(self):
        with open(self.label_dict_path, 'r') as f:
            for line in f:
                name, label = line.strip('\n').split(',')
                self.label_dict[name] = self.char_dict[label][0]

    def set_train_label(self):
        with open(self.train_set_path, 'r') as f:
            sentence_dict = {}
            for line in f:
                name, *features = line.strip('\n').split(' ')
                spkird, sentid, timeid = name.split('_')
                key = "%s_%s" % (spkird, sentid)
                if key not in sentence_dict:
                    sentence_dict[key] = Sentence(key)
                sentence_dict[key].feature[int(timeid)-1][:] = np.array(features).astype(np.float)
                sentence_dict[key].label[int(timeid)-1][self.label_dict[name]] = 1.0
        
        all_feature = []
        all_label = []
        for pair in sentence_dict.items():
            all_feature.append([pair[1].feature])
            all_label.append([pair[1].label])
        
        return (np.vstack(all_feature), np.vstack(all_label))

    def set_reduce_dict(self):
        with open(self.reduce_dict_path) as f:
            for line in f:
                original, convert = line.strip('\n').split('\t')
                self.reduce_dict[original] = convert

class Sentence(object):
    def __init__(self, name, feature_width=69, max_length=800, char_variety=48):
        self.name = name
        self.feature_width = feature_width
        self.max_length = max_length
        self.feature = np.zeros((max_length, feature_width))
        self.label = np.zeros((max_length, char_variety))