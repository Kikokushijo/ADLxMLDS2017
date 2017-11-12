import os
import json
import sys
import numpy as np

max_dict_length = 4000
max_sent_length = 50

class Label(object):
    def __init__(self, dictionary):
        self.mv2featdict = dict()
        self.word2intdict = dict()
        self.int2worddict = dict()
        self.mv2sentdict = dict()
        self.mvlist = []
        self.set_dict(dictionary, need_reduce=True)
        
    def set_dict(self, dictionary, need_reduce=True):
        try:
            with open('dict/word2int_%d.json' % (max_dict_length), 'r') as f:
                self.word2intdict = json.load(f)

            with open('dict/int2word_%d.json' % (max_dict_length), 'r') as f:
                self.int2worddict = json.load(f)
                
            for entries in dictionary:
                self.mv2sentdict[entries['id']] = entries['caption']
                
        except FileNotFoundError:
            
            print('NoDictFound')
            
            words_frequency = dict()
            for entries in dictionary:
                self.mv2sentdict[entries['id']] = entries['caption']
                sentences = []
                for sentence in entries['caption']:
                    words = sentence.split(' ')
                    for word in words:
                        word = word.lower().strip('!., ')
                        if not word:
                            break
                        if word not in words_frequency:
                            words_frequency[word] = 1
                        else:
                            words_frequency[word] += 1
            if need_reduce:
                freqs = sorted(list(words_frequency.items()), reverse=True, key=lambda x:x[1])[:max_dict_length-3]
                words_frequency = {key:freq for key, freq in freqs}

            m = max_dict_length
            self.word2intdict = {'<unk>':m-3, '<start>':m-2, '<end>':m-1}
            self.int2worddict = {m-3:'<unk>', m-2:'<start>', m-1:'<end>'}
            for index, key in enumerate(words_frequency):
                self.word2intdict[key] = index
                self.int2worddict[index] = key
            
            with open('dict/word2int_%d.json' % (max_dict_length), 'w') as f:
                json.dump(self.word2intdict, f)

            with open('dict/int2word_%d.json' % (max_dict_length), 'w') as f:
                json.dump(self.int2worddict, f)
    
    def word2int(self, word):
        return self.word2intdict[word]
    
    def int2word(self, integer):
        return self.int2worddict[str(integer)]
    
    def mv2feat(self, moviename):
        return self.mv2featdict[moviename]
    
    def mv2sent(self, moviename):
        return self.mv2sentdict[moviename]
    
    def set_feat(self, features, filename):
        self.mvlist.append(filename)
        self.mv2featdict[filename] = features
    
    def sent2onehot(self, sent):
        words = self.sent2words(sent)
        index = self.words2index(words)
        onehot = self.index2onehot(index)
        return onehot
    
    def sent2words(self, sent):
        words = sent.split(' ')
        return [word.lower().strip('!., ') for word in words]
    
    def words2index(self, words):
        m = max_dict_length
        n = max_sent_length
        return [m-2] + [self.word2intdict[word] if word in self.word2intdict else m-3 for word in words] + [m-1] * (n - 1 - len(words))
#         return [m-2] + [self.word2intdict[word] if word in self.word2intdict else m-3 for word in words] + [m-1]
        
    def index2onehot(self, indices):
        indices = np.array(indices)
        onehot = np.zeros((len(indices), max_dict_length))
        onehot[np.arange(len(indices)), indices] = 1
        return onehot

class Dataset(object):
    def __init__(self, feat_dirname=None, json_filename=None):
        if not feat_dirname:
            self.feat_dirname = os.path.join(sys.argv[1], 'training_data/feat/')
        else:
            self.feat_dirname = feat_dirname
        if not json_filename:
            self.json_filename = os.path.join(sys.argv[1], 'training_label.json')
        else:
            self.json_filename = json_filename
        self.read_label()
        self.read_feats()
    
    def read_label(self):
        with open(self.json_filename, 'r') as f:
            self.label = Label(json.load(f))
    
    def read_feats(self):
        movies = []
        for filename in os.listdir(self.feat_dirname):
            if filename.endswith('.npy'):
                with open(os.path.join(self.feat_dirname, filename), 'rb') as f:
                    self.label.set_feat(np.load(f), filename[:-4])