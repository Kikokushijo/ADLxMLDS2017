from agent_dir.agent import Agent
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

import numpy as np
import scipy

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)
        
        self.state_size = 6400
        self.episode = 0
        self.max_episode = 1000
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.action_size = self.env.env.action_space.n
        self.gamma = 0.995
        self.learning_rate = 0.01
        self.model = self.build_model()
        self.model.load_weights('Pong.h5')

        if args.test_pg:
            #you can load your model here
            # self.model.load_weights(model_path)
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################

        self.state = self.env.reset()
        self.prev_x = None
        self.score = 0
        self.battle_score = 0

    def act(self, state):
        state = state.reshape([1, 6400])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def preprocess(self, o, image_size=[80,80]):
        """
        Call this function to preprocess RGB image to grayscale image if necessary
        This preprocessing code is from
            https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
        
        Input: 
        RGB image: np.array
            RGB screen of game, shape: (210, 160, 3)
        Default return: np.array 
            Grayscale image, shape: (80, 80, 1)
        
        """
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, image_size)
        return np.expand_dims(resized.astype(np.float32),axis=2)

    def train(self):
        """
        Implement your training algorithm here
        """

        ##################
        # YOUR CODE HERE #
        ##################
        bscore = []
        while self.episode < self.max_episode:
            self.init_game_setting()
            self.episode += 1

            done = False
            while not done:
                self.cur_x = self.preprocess(self.state)
                x = self.cur_x - self.prev_x if self.prev_x is not None else np.zeros((80, 80, 1))
                self.prev_x = self.cur_x
                
                action, prob = self.act(x)
                state, reward, done, info = self.env.step(action)
                self.score += reward
                self.remember(x, action, prob, reward)

            print('Episode: %d' % self.episode)
            gradients = np.vstack(self.gradients)
            rewards = np.vstack(self.rewards)
            rewards = self.discount_rewards(rewards)
            rewards = rewards / np.std(rewards - np.mean(rewards))
            gradients *= rewards
            # print(len(self.states))
            # for state in self.states:
            #     print(state.shape)
            X = np.squeeze(np.vstack([self.states])).reshape(-1, 6400)
            # X = np.vstack(self.states)
            # print('X Shape:', X.shape)
            Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
            # print(X.shape, Y.shape)
            # print(X[0].shape)
            self.model.train_on_batch(np.asarray(X), Y)
            self.model.save_weights('Pong.h5')
            print('Battle Score:', self.battle_score)
            bscore.append(self.battle_score)
            self.states, self.probs, self.gradients, self.rewards = [], [], [], []
            with open('average.csv', 'a') as f:
                f.write('%d, %d\n' %(self.episode, self.battle_score))

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        # print('State Shape:', state.shape)
        self.states.append(state)
        # print('States Shape:', np.asarray(self.states).shape)
        self.rewards.append(reward)
        # print('Reward:', reward)
        if reward == 1.0:
            self.battle_score += 1

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        if test:
            action, prob = self.act(observation)
            return action

    def build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size, )))
        model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
                                activation='selu', init='lecun_normal'))
        model.add(Flatten())
        model.add(Dense(64, activation='selu', init='lecun_normal'))
        model.add(Dense(32, activation='selu', init='lecun_normal'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.summary()
        return model

        # model = Sequential()
        # model.add(Convolution2D(2, 3, 3, input_shape=(1, I, J), border_mode='valid'))
        # model.add(Flatten())
        # model.add(Dropout(0.1))
        # model.add(Dense(env.action_space.n, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')