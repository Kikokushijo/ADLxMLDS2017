from agent_dir.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
# from torch.distributions import Categorical

import scipy.misc
import numpy as np

from itertools import count
from copy import copy
import os

useGPU = torch.cuda.is_available()


def prepro(o, image_size=[80,80]):
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

    # TA's preprocessing method

    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)

        # hyperparameter

        self.gamma = 0.99
        self.batch_size = 1
        self.lr = 1e-4
        self.decay_rate = 0.99
        self.state = None

        # build model

        self.policy_network = PolicyNetwork()


        # CPU vs GPU
        if useGPU:
            self.policy_network = self.policy_network.cuda()

        if os.path.isfile('PG.pkl'):
            print('loading trained model')
            self.policy_network.load_state_dict(torch.load('PG.pkl'))
        self.opt = optim.RMSprop(self.policy_network.parameters(), lr=self.lr, weight_decay=self.decay_rate)

        # log

        # self.rw_log = open('pg_record/pg_CNN.csv', 'a')

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """

        self.state = None

        pass

    def update_param(self):

        accumulated = 0
        rewards = []        
        for r in reversed(self.policy_network.rewards):
            if r != 0:
                accumulated = 0
            accumulated = r + self.gamma * accumulated
            rewards.append(accumulated)
        rewards.reverse()

        print('Total Action Steps', len(rewards))

        # normalize

        if useGPU:
            rewards = torch.Tensor(rewards).cuda()
        else:
            rewards = torch.Tensor(rewards)

        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # calculate grad(J)

        policy_loss = []
        for log_prob, reward in zip(self.policy_network.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        # theta = theta + alpha * grad(J)

        self.opt.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.opt.step()

        del self.policy_network.rewards[:]
        del self.policy_network.saved_log_probs[:]

    def train(self):
        """
        Implement your training algorithm here
        """

        self.eps_reward = 0
        self.avg_reward = -21
        self.rewards = []
        for num_episode in count(1):
            state = self.env.reset()
            self.init_game_setting()
            for t in range(10500):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                self.eps_reward += reward
                self.policy_network.rewards.append(reward)

                if done:
                    self.avg_reward = 0.98 * self.avg_reward + 0.02 * self.eps_reward
                    print('Episode: %d, Eps-reward: %f. Running-reward: %f' % (num_episode, self.eps_reward, self.avg_reward))
                    self.rw_log.write('%d, %f, %f\n' % (num_episode, self.eps_reward, self.avg_reward))
                    self.rewards.append(self.eps_reward)
                    self.eps_reward = 0
                    

                    if num_episode % 30 == 0:
                        print('Recent 30 Episode: %f' % (sum(self.rewards) / 30))
                        self.rewards = []
                    break

            if num_episode % self.batch_size == 0:
                self.update_param()

            if num_episode % 50 == 0:
                print('Save model')
                torch.save(self.policy_network.state_dict(), 'PG.pkl')
                print('Complete Saving Model')
                pass


    def make_action(self, state, test=False):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        state = prepro(state)
        if self.state is None:
            self.state = state
            state_dif = state
        else:
            state_dif = self.state - state

        self.state = state
        tmp = torch.from_numpy(state_dif).float().unsqueeze(0)

        # CPU vs GPU
        if useGPU:
            probs = self.policy_network(Variable(tmp).cuda())
        else:
            probs = self.policy_network(Variable(tmp))
        if test:
            prob, action = torch.max(probs, 1)
            return action.data[0] + 1
        else:
            
            m = Categorical(probs)
            action = m.sample()
            self.policy_network.saved_log_probs.append(m.log_prob(action))
            return action.data[0] + 1

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=4,
                      stride=2),
            nn.ReLU(),
        )

        self.affine1 = nn.Linear(32 * 8 * 8, 128)
        self.affine2 = nn.Linear(128, 3)

        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        x = self.conv1(x.view(-1, 1, 80, 80))
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        return F.softmax(self.affine2(x))

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if useGPU:
            super(Variable, self).__init__(data.cuda(), *args, **kwargs)
        else:
            super(Variable, self).__init__(data.cuda(), *args, **kwargs)
