import read_data as rd
import bleu_eval as be

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import RMSprop, Adam
from keras.layers.wrappers import TimeDistributed

import random
import numpy as np
import os
import sys

num_encoder_tokens = 4096
num_densed_tokens = 1024
latent_dim = 2048
num_decoder_tokens = rd.max_dict_length
max_dict_length = rd.max_dict_length
max_sent_length = rd.max_sent_length

def set_model(mode='default'):

    if mode == 'default':
        # CNN_inputs = Input(shape=(None, num_encoder_tokens))
        # denser = TimeDistributed(Dense(num_densed_tokens, activation='relu'))
        # dense_outputs = denser(CNN_inputs)

        # encoder_inputs = Input(shape=(None, num_densed_tokens))
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True, implementation=2)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, implementation=2)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        # model = Model([CNN_inputs, decoder_inputs], decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')
        print(model.summary())

        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return model, encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model):

    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, label.word2int('<start>')] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = label.int2word(int(sampled_token_index))
        decoded_sentence += sampled_word

        if sampled_word == '<end>':
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
        decoded_sentence += ' ' 

    return decoded_sentence

def output_caption_txt(e, d, movies=None):
    if not movies:
        movies = test_DS.label.mvlist
    with open(sys.argv[2], 'w') as f:
        for index, movie in enumerate(movies):
            input_seq = test_DS.label.mv2feat(movie)
            decoded_sentence = decode_sequence(input_seq.reshape(1, 80, 4096), e, d)
            tmp = (decoded_sentence[:-7] + '.').capitalize()
            print(movie + ',' + tmp)
            f.write(movie + ',' + tmp + '\n')

if __name__ == '__main__':
    feat_path = os.path.join(sys.argv[1], 'testing_data/feat/')
    test_DS = rd.Dataset(feat_path, None)
    label = test_DS.label
    model, e, d = set_model()
    model.load_weights('models/1110_random2.h5')
    movies = ['klteYv1Uv9A_27_33.avi', 
              '5YJaS2Eswg0_22_26.avi',
              'UbmZAe5u5FI_132_141.avi',
              'JntMAcTlOF0_50_70.avi',
              'tJHUH9tpqPg_113_118.avi']
    output_caption_txt(e, d, movies)