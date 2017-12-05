from agent_dir.agent import Agent

import torch.nn as nn
import torch.optim as optim
import torch
import os

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.num_actions = env.action_space.n
        self.replay_buffer_size = 1000000
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_starts = 50000
        self.learning_freq = 4
        self.frame_history_len = 4
        self.target_update_freq = 10000
        self.build_model()

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

    def build_model(self):
        self.Q = MyCNN(self.num_actions).type(torch.cuda.FloatTensor)
        self.target_Q = MyCNN(self.num_actions).type(torch.cuda.FloatTensor)
        if os.path.isfile('DQN_Q.pkl'):
            print('Load Q parametets ...')
            self.Q.load_state_dict(torch.load('DQN_Q.pkl'))
        else:
            print('Start Training From Scratch')
        
        if os.path.isfile('DQN_targetQ.pkl'):
            print('Load target Q parameters ...')
            self.target_Q.load_state_dict(torch.load('DQN_targetQ.pkl'))
        else:
            print('Start Training From Scratch')

        self.opt = optim.RMSprop(self.Q.parameters(), lr=0.005, alpha=0.95, eps=0.01)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################

        self.state = self.env.reset()

        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        env = self.env

        # print(env.observation_space.shape)
        

        while True:
            self.init_game_setting()
            print('START NEW GAME!')
            done = False
            while not done:
                self.state, reward, done, info = self.env.step(self.make_action(self.state))
                print('Reward:', reward)


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()

class MyCNN(nn.Module):
    def __init__(self, num_actions):

        super(MyCNN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(in_channels=4, 
                                             out_channels=32, 
                                             kernel_size=8, 
                                             stride=1, 
                                             padding=1))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(4, 4))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(in_channels=32, 
                                             out_channels=64, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(in_channels=64, 
                                             out_channels=128, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, num_actions))
        self.layer4 = layer4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fc_input = x.view(x.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

class Buffer(object):
    def __init__(self, buffer_size, history_len):
        self.buffer_size = buffer_size
        self.history_len = history_len

    def store(self, frame):
        pass