import read_data as rd
import bleu_eval as be

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import RMSprop, Adam
from keras.layers.wrappers import TimeDistributed

import random
import numpy as np

num_encoder_tokens = 4096
num_densed_tokens = 1024
latent_dim = 2048
num_decoder_tokens = rd.max_dict_length
max_dict_length = rd.max_dict_length
max_sent_length = rd.max_sent_length

def input_generator(batch_size=64, option='all_sentence'):
    if option == 'all_sentence':
        total_movies = len(label.mvlist)
        curr_movie = 0
        curr_sent = 0
        while True:
            feat = np.array([]).reshape(0, 80, 4096)
            sent = np.array([]).reshape(0, max_sent_length, max_dict_length)
            for i in range(batch_size):
                if curr_sent >= len(label.mv2sent(label.mvlist[curr_movie % total_movies])):
                    curr_sent -= len(label.mv2sent(label.mvlist[curr_movie % total_movies]))
                # if curr_sent >= 1:
                #     curr_sent -= 1
                    curr_movie += 1
                feat = np.concatenate((feat, [label.mv2feat(label.mvlist[curr_movie % total_movies])]))
                sent = np.concatenate((sent, [label.sent2onehot(label.mv2sent(label.mvlist[curr_movie % total_movies])[curr_sent])]))
                curr_sent += 1
                
            yield [[feat, sent[:, :-1, :]], sent[:, 1:, :]]
    if option == 'random_each':
        total_movies = len(label.mvlist)
        curr_movie = 0
        while True:
            feat = np.array([]).reshape(0, 80, 4096)
            sent = np.array([]).reshape(0, max_sent_length, max_dict_length)
            for i in range(batch_size):
                # if curr_sent >= len(label.mv2sent(label.mvlist[curr_movie % total_movies])):
                #     curr_sent -= len(label.mv2sent(label.mvlist[curr_movie % total_movies]))
                # if curr_sent >= 1:
                #     curr_sent -= 1
                #     curr_movie += 1
                feat = np.concatenate((feat, [label.mv2feat(label.mvlist[curr_movie % total_movies])]))
                curr_sent = random.randint(0, len(label.mv2sent(label.mvlist[curr_movie % total_movies])) - 1)
                sent = np.concatenate((sent, [label.sent2onehot(label.mv2sent(label.mvlist[curr_movie % total_movies])[curr_sent])]))
                curr_movie += 1
                
            yield [[feat, sent[:, :-1, :]], sent[:, 1:, :]]

def set_model(mode='default'):

    if mode == 'default'
        # CNN_inputs = Input(shape=(None, num_encoder_tokens))
        # denser = TimeDistributed(Dense(num_densed_tokens, activation='relu'))
        # dense_outputs = denser(CNN_inputs)

        # encoder_inputs = Input(shape=(None, num_densed_tokens))
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True, implementation=2)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # encoder_outputs, state_h, state_c = encoder(dense_outputs)
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

    # elif mode == 'GRU':
    #     encoder_inputs = Input(shape=(None, num_encoder_tokens))
    #     encoder = GRU(latent_dim, return_state=True, implementation=2)
    #     encoder_outputs, state_h = encoder(encoder_inputs)

    #     decoder_inputs = Input(shape=(None, num_decoder_tokens))
    #     decoder_gru = GRU(latent_dim, return_sequences=True)
    #     decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    #     decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    #     decoder_outputs = decoder_dense(decoder_outputs)
    #     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #     model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')

    #     encoder_model = Model(encoder_inputs, encoder_states)
    #     decoder_state_input = Input(shape=(latent_dim,))
    #     # decoder_state_input_c = Input(shape=(latent_dim,))
    #     # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    #     decoder_outputs, decoder_state = decoder_gru(decoder_inputs, initial_state=decoder_state_inputs)
    #     # decoder_states = [state_h, state_c]
    #     decoder_outputs = decoder_dense(decoder_outputs)
    #     decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_state)

    #     return model, encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, label.word2int('<start>')] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
#         print(output_tokens[0, -1, 73])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = label.int2word(int(sampled_token_index))
        decoded_sentence += sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_word == '<end>':
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
        decoded_sentence += ' ' 

    return decoded_sentence

def output_decoded_sequence(e, d, test=10):
    movies = label.mvlist
    for index, movie in enumerate(movies[:test]):
        input_seq = label.mv2feat(movie)
        decoded_sentence = decode_sequence(input_seq.reshape(1, 80, 4096), e, d)
        print('Decoded sentence: ', decoded_sentence)

def output_caption_txt(e, d):
    movies = test_DS.label.mvlist
    # with open('caption.txt', 'w') as f:

    result = {}
    for index, movie in enumerate(movies):
        input_seq = test_DS.label.mv2feat(movie)
        decoded_sentence = decode_sequence(input_seq.reshape(1, 80, 4096), e, d)
#         print('Video Name: ', movie)
#         print('Decoded Sentence: ', decoded_sentence)
        tmp = (decoded_sentence[:-7] + '.').capitalize()
        result[movie] = tmp
        # print(movie + ',' + tmp)
        # for sent in label.mv2sent(movie):
        #     print(sent)
        # f.write(movie + ',' + tmp + '\n')
    return result

def validation(result):
    return be.val_BLEU(result)

if __name__ == "__main__":
    DS = rd.Dataset()
    test_DS = rd.Dataset('../hw2_dataset/testing_data/feat/', '../hw2_dataset/testing_label.json')
    label = DS.label
    gen = input_generator(option='random_each')
    model_filename = 'models/1110_random2.h5'

    model, e, d = set_model()
    max_score = 0

    # model.load_weights(model_filename)
    for i in range(150):
        model.fit_generator(gen, 4, epochs=50, verbose=1)
        # model.save_weights(model_filename)
        output_decoded_sequence(e, d)
        score = validation(output_caption_txt(e, d))
        if score > max_score:
            max_score = score
            model.save_weights(model_filename)
        print('This Epoch:', score, 'Current Max:', max_score)
        # validation()