from agent_dir.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Categorical

import scipy.misc
import numpy as np

from itertools import count
import os

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

    # Another preprocessing method

    o = o[35:195]
    o = o[::2, ::2, 0]
    o[o == 144] = 0
    o[o == 109] = 0
    o[o != 0 ] = 1
    return o.astype(np.float).reshape(-1)


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

        # build model

        self.policy_network = PolicyNetwork()
        if os.path.isfile('pg_record/pg.pkl'):
            print('loading trained model')
            self.policy_network.load_state_dict(torch.load('pg_record/pg.pkl'))
        self.opt = optim.RMSprop(self.policy_network.parameters(), lr=self.lr, weight_decay=self.decay_rate)

        # log

        self.rw_log = open('pg_record/pg_record.csv', 'a')

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def update_param(self):

        accumulated = 0
        rewards = []        
        for r in reversed(self.policy_network.rewards):
            accumulated = r + self.gamma * accumulated
            rewards.append(accumulated)
        rewards.reverse()

        # normalize

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

        self.policy_network.rewards = []
        self.policy_network.saved_log_probs = []

    def train(self):
        """
        Implement your training algorithm here
        """

        self.eps_reward = 0
        for num_episode in count(1):
            state = self.env.reset()
            for t in range(10000):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                self.eps_reward += reward
                self.policy_network.rewards.append(reward)

                if done:
                    print('Eps-reward: %f.' % (self.eps_reward))
                    # self.rw_log.write('%d, %f\n' % (num_episode, self.eps_reward))
                    self.eps_reward = 0
                    break

            if num_episode % self.batch_size == 0:
                self.update_param()

            if num_episode % 50 == 0:
                # torch.save(self.policy_network.state_dict(), 'pq_record/pg.pkl')
                pass


    def make_action(self, state, test=True):
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
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.policy_network.saved_log_probs.append(m.log_prob(action))
        return action.data[0] + 1

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 16, 5, 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 32, 5, 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.affine1 = nn.Linear(32 * 20 * 20, 3)

        self.affine1 = nn.Linear(6400, 64)
        self.affine2 = nn.Linear(64, 3)
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        # x = self.conv1(x.view(-1, 1, 80, 80))
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)
        # action_scores = self.affine1(x)
        return F.softmax(action_scores, dim=1)

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data.cuda(), *args, **kwargs)