import numpy as np

class Dataset(object):
    def __init__(self, ldp='train.lab', tsp='fbank/train_copy.ark',
                       cdp='48phone_char.map', rp='48_39.map'):

        self.char_dict = {}
        self.label_dict = {}
        self.reduce_dict = {}
        self.sentence = Sentence()
        
        self.char_dict_path = cdp
        self.set_char_dict()

        self.label_dict_path = ldp
        self.set_label_dict()

        self.reduce_dict_path = rp
        self.set_reduce_dict()

        self.train_set_path = tsp

    def set_char_dict(self):
        with open(self.char_dict_path) as f:
            for line in f.readlines():
                phone, num, char = line.strip('\n').split('\t')
                self.char_dict[phone] = [num, char]

    def set_label_dict(self):
        with open(self.label_dict_path, 'r') as f:
            for line in f:
                name, label = line.strip('\n').split(',')
                self.label_dict[name] = self.char_dict[label][0]

    def set_train_label(self):
        while True:
            with open(self.train_set_path, 'r') as f:
                for line in f:
                    name, *features = line.strip('\n').split(' ')
                    spkrid, sentid, timeid = name.split('_')
                    print(name)

                    if  (spkrid != self.sentence.spkrid or
                        sentid != self.sentence.sentid):
                        if self.sentence.spkrid is not None:
                            X_train = np.array(self.sentence.timestep_features).astype(np.float)
                            X_train = np.lib.pad(X_train, ((0, 800-len(X_train)), (0, 0)), 'constant', constant_values=0)
                            Y_train = np.array(self.sentence.timestep_label).astype(int)
                            yield (X_train, Y_train)
                        self.sentence = Sentence(spkrid, sentid)
                    self.sentence.append_timestep(features, self.label_dict[name])

    def set_reduce_dict(self):
        with open(self.reduce_dict_path) as f:
            for line in f:
                original, convert = line.strip('\n').split('\t')
                self.reduce_dict[original] = convert

class Sentence(object):
    def __init__(self, spkrid=None, sentid=None):
        self.spkrid = spkrid
        self.sentid = sentid
        self.timestep_features = np.array([])
        self.timestep_label = np.array([])

    def append_timestep(self, feature, label):
        self.timestep_features = np.append(self.timestep_features, feature)
        self.timestep_label = np.append(self.timestep_label, label)

class Sentence_Batch(object):
    def __init__(self):
        pass

if __name__ == "__main__":
    DS = Dataset()
    data_gen = DS.set_train_label()
    for i in range(3):
        X, Y = next(data_gen)


