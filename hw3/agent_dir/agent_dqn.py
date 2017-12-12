from agent_dir.agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from collections import namedtuple
from itertools import count
import random
import os

use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        
        # hyperparameter

        self.buffer_size = 10000
        self.max_step = 10000000
        self.start_step = 10000
        self.update_target_step = 1000
        self.update_online_step = 4
        self.gamma = 0.99
        self.batch_size = 32
        self.schedule = ExplorationSchedule()

        # set replay buffer

        self.buffer = ReplayMemory(self.buffer_size)

        # build Q networks

        self.num_actions = env.action_space.n
        self.Q = DQN(self.num_actions)
        self.Q_target = DQN(self.num_actions)
        self.opt = optim.RMSprop(self.Q.parameters(), lr=1e-4)
        if use_cuda:
            self.Q.cuda()
            self.Q_target.cuda()
        if os.path.isfile('dqn_record/Q.pkl'):
            print('loading trained model')
            self.Q.load_state_dict(torch.load('dqn_record/Q.pkl'))
            self.Q_target.load_state_dict(self.Q.state_dict())



        # initialize

        self.time = 0
        
        if args.test_dqn:
            #you can load your model here
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
        pass

    def optimize_model(self):
        if len(self.buffer) < self.start_step:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
        # print(np.array(batch.state).shape)
        non_final_next_states = Variable(torch.cat(Tensor(np.asarray([[s] for s in batch.next_state
                                                if s is not None]))), volatile=True)
        # print([s for s in batch.next_state
        #                                         if s is not None])
        state_batch = Variable(torch.cat(Tensor(np.array([batch.state]))))
        action_batch = Variable(torch.cat(LongTensor(np.array([batch.action]))))
        reward_batch = Variable(torch.cat(Tensor(np.array([batch.reward]))))
        # print(self.Q(state_batch))
        # print(action_batch.unsqueeze(0))
        state_action_values = self.Q(state_batch).gather(1, action_batch.view(-1, 1))

        next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.Q_target(non_final_next_states).max(1)[0]
        next_state_values.volatile = False
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.opt.zero_grad()
        loss.backward()
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

    def train(self):
        """
        Implement your training algorithm here
        """
        
        self.rewards = []
        for num_episode in count(1):
            state = self.env.reset()
            done = False
            self.eps_reward = 0
            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, _ = self.env.step(action)
                self.eps_reward += reward
                self.time += 1

                self.buffer.push(state, action, next_state, reward)
                state = next_state

                if self.time % self.update_online_step == 0:
                    self.optimize_model()

                if self.time % self.update_target_step == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                    self.

                if self.time % 1000 == 0:
                    print('Now playing %d steps.' % (self.time))

            self.rewards.append(self.eps_reward)
            if num_episode % 100 == 0:
                print('Recent 100 episode: %f' % (sum(self.rewards) / 100))
                self.rewards = []
            


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
        # obs = np.transpose(observation, (2, 0, 1))
        if not test:
            rd = random.random()
            eps = self.schedule.value(self.time)
            if rd > eps:
                obs = torch.from_numpy(np.array([observation]))
                return int(self.Q(Variable(obs, volatile=True).type(Tensor)).data.max(1)[1].view(1, 1))
            else:
                return int(LongTensor([[random.randrange(self.num_actions)]]))

        return self.env.get_random_action()

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, num_actions):

        super(DQN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(in_channels=4, 
                                             out_channels=32, 
                                             kernel_size=8, 
                                             stride=4))
        layer1.add_module('relu1', nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(in_channels=32, 
                                             out_channels=64, 
                                             kernel_size=4, 
                                             stride=2))
        layer2.add_module('relu2', nn.ReLU(True))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(in_channels=64, 
                                             out_channels=64, 
                                             kernel_size=3, 
                                             stride=1))
        layer3.add_module('relu3', nn.ReLU(True))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(64 * 7 * 7, 512))
        layer4.add_module('lrelu1', nn.LeakyReLU(True))
        layer4.add_module('fc2', nn.Linear(512, num_actions))
        self.layer4 = layer4

    def forward(self, x):
        # print(type(x))
        # print(x.shape)
        x = x.permute(0, 3, 1, 2).cuda()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        fc_input = x.view(x.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

class ExplorationSchedule(object):
    def __init__(self, timestep=1e6, final=0.95, initial=0):
        self.timestep = timestep
        self.final = final
        self.initial = initial

    def value(self, t):
        return self.initial + (self.final - self.initial) * min(t / self.timestep, 1.0)