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
        self.start_step = 10000
        self.max_step = 10000000
        self.learning_rate = 1e-4
        self.update_target_step = 1000
        self.update_online_step = 4
        self.gamma = 0.99
        self.batch_size = 32
        self.num_actions = env.action_space.n

        # set explore schedule

        self.schedule = ExplorationSchedule()

        # set replay buffer

        self.buffer = ReplayMemory(self.buffer_size)

        # build Q networks


        self.Q = DQN(self.num_actions)
        self.Q_target = DQN(self.num_actions)
        self.opt = optim.RMSprop(self.Q.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        if use_cuda:
            self.Q = self.Q.cuda()
            self.Q_target = self.Q_target.cuda()
            self.loss_func = self.loss_func.cuda()
        if os.path.isfile('dqn_record/double_DQN.pkl'):
            print('loading trained model')
            self.Q.load_state_dict(torch.load('dqn_record/double_DQN.pkl'))
            self.Q_target.load_state_dict(self.Q.state_dict())

        # initialize

        self.time = 0

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def optimize_model(self):
        if len(self.buffer) < self.start_step:
            return

        b_memory = self.buffer.sample()
        b_s, b_a, b_r, b_s_, b_nd = [], [], [], [], []
        for s, a, r, s_, d in b_memory:
            b_s.append(s)
            b_a.append(a)
            b_r.append(r)
            b_s_.append(s_)
            b_nd.append(d)
        # print(b_nd)

        b_s = Variable(torch.FloatTensor(np.array(b_s)).transpose(1, 3))
        b_a = Variable(torch.LongTensor(np.array(b_a)).unsqueeze(1))
        b_r = Variable(torch.FloatTensor(np.array(b_r)).unsqueeze(1))
        b_s_ = Variable(torch.FloatTensor(np.array(b_s_)).transpose(1, 3))
        b_nd = Variable(torch.Tensor(b_nd).unsqueeze(1))
        # print(b_nd)

        if use_cuda:
            b_s = b_s.cuda()
            b_a = b_a.cuda()
            b_r = b_r.cuda()
            b_s_ = b_s_.cuda()
            b_nd = b_nd.cuda()

        q_eval = self.Q(b_s).gather(1, b_a)
        q_next = self.Q_target(b_s_).detach()
        q_next.volatile = False


        values, indices = torch.max(self.Q(b_s_), 1)
        q_true = q_next.gather(1, indices.unsqueeze(1))
        
        # q_target = b_r + self.gamma * q_next.max(1, keepdim=True)[0] * b_nd
        q_target = b_r + self.gamma * q_true * b_nd

        loss = self.loss_func(q_eval, q_target)
        self.opt.zero_grad()
        loss.backward()
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

                self.buffer.push(state, action, reward, next_state, not done)
                state = next_state

                if self.time % self.update_online_step == 0:
                    self.optimize_model()

                if self.time % self.update_target_step == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                    torch.save(self.Q.state_dict(), 'dqn_record/double_DQN.pkl')

                if self.time % 1000 == 0:
                    print('Now playing %d steps.' % (self.time))

            print('Step: %d, Episode: %d, Episode Reward: %f' % (self.time, num_episode, int(self.eps_reward)))

            with open('dqn_record/double_DQN.csv', 'a') as f:
                f.write("%d, %d\n" % (self.time, self.eps_reward))
            
            self.rewards.append(self.eps_reward)
            if num_episode % 100 == 0:
                print('---')
                print('Recent 100 episode: %f' % (sum(self.rewards) / 100))
                print('---')
                self.rewards = []

            if self.time >= 3000000:
                break
            


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
        obs = Variable(torch.from_numpy(observation).unsqueeze(0).transpose(1, 3))
        if use_cuda:
            obs = obs.cuda()

        if not test:
            rd = random.random()
            eps = self.schedule.value(self.time)
            if rd < eps:
                return self.Q(obs).data.max(1)[1][0]
            else:
                return self.env.get_random_action()
        else:
            return self.Q(obs).data.max(1)[1][0]

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, s, a, r, s_, d):
        if self.position < self.capacity:
            self.memory.append(None)
        transition = (s, a, r, s_, d)
        index = self.position % self.capacity
        self.memory[index] = transition
        self.position += 1

    def sample(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, num_actions):

        super(DQN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, 
                      out_channels=32, 
                      kernel_size=8, 
                      stride=4),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=4, 
                      stride=2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1),
            nn.ReLU(),
        )

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(64 * 7 * 7, 512))
        layer4.add_module('lrelu1', nn.LeakyReLU())
        layer4.add_module('fc2', nn.Linear(512, num_actions))
        self.layer4 = layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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