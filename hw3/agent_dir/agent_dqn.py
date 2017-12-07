from agent_dir.agent import Agent
from model.torch_models import DQNCNN, SimpleDQN
from util.schedule import LinearSchedule
from util.variable import Variable
from util.buffer import Buffer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import numpy as np
import random
import os

dtype = torch.cuda.DoubleTensor

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.num_actions = env.action_space.n
        self.schedule = LinearSchedule()
        self.Q_filename = os.path.join('model', 'DQN_Q.pkl')
        self.targetQ_filename = os.path.join('model', 'DQN_targetQ.pkl')
        self.build_model()
        self.buffer_size = int(8e4)
        self.buffer = Buffer(self.buffer_size)
        self.gamma = 0.99
        self.training_record = os.path.join('record', 'DQN.csv')

        self.loss_func = nn.MSELoss()

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        else:
            pass
        

    def build_model(self):
        # self.Q = DQNCNN(self.num_actions).type(dtype)
        # self.target_Q = DQNCNN(self.num_actions).type(dtype)
        self.Q = SimpleDQN(self.num_actions).type(dtype)
        self.target_Q = SimpleDQN(self.num_actions).type(dtype)
        if os.path.isfile(self.Q_filename):
            print('Load Q parametets ...')
            self.Q.load_state_dict(torch.load(self.Q_filename))
        else:
            print('Start Training From Scratch')
        
        if os.path.isfile(self.targetQ_filename):
            print('Load target Q parameters ...')
            self.target_Q.load_state_dict(torch.load(self.targetQ_filename))
        else:
            print('Start Training From Scratch')

        self.opt = optim.RMSprop(self.Q.parameters(), lr=0.00001, alpha=0.95, eps=0.01)

    def save_model(self):
        torch.save(self.Q.state_dict(), self.Q_filename)
        torch.save(self.target_Q.state_dict(), self.targetQ_filename)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """

        self.state = self.env.reset()

    def select_training_action(self, state, t):
        rd = random.random()
        eps = self.schedule.value(t)
        if rd > eps:
            # print(state.shape)
            state = torch.from_numpy(np.transpose(state, (2, 0, 1))).type(dtype).unsqueeze(0) / 255.0
            # print(state.shape)
            # print(self.Q(Variable(state, volatile=True)).data.max(1)[1].cpu())
            return self.Q(Variable(state, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(self.num_actions)]])

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        from itertools import count

        self.init_game_setting()

        num_updates = 0
        target_update_freq = 10000
        num_episode = 0
        episode_reward = 0
        episode_rewards = []

        for t in count():

            if t % 100 == 0:
                print('Step %d' %(t))

            if t > self.buffer_size:
                # action = self.select_training_action(self.state, t)[0, 0]
                action = int(self.select_training_action(self.state, t)[0])
            else:
                action = int(random.randrange(self.num_actions))

            # print('Action:', action)

            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            reward = max(-1.0, min(reward, 1.0))
            self.buffer.append(self.state, action, reward, next_state, done)
            self.state = next_state

            if done:
                self.init_game_setting()
                num_episode += 1
                episode_rewards.append(episode_reward)
                episode_reward = 0
                if t > self.buffer_size:
                    with open(self.training_record, 'a') as f:
                        f.write('%d, %d\n' %(num_episode, episode_rewards[-1]))

            if t > self.buffer_size:
                s_batch, a_batch, r_batch, ns_batch, done_mask = self.buffer.sample()
                
                s_batch = Variable(torch.from_numpy(np.asarray(np.transpose(s_batch, (0, 3, 1, 2)))).type(dtype) / 255.0)
                a_batch = Variable(torch.from_numpy(np.asarray(a_batch)).long()).cuda()
                r_batch = Variable(torch.from_numpy(np.asarray(r_batch))).cuda()
                ns_batch = Variable(torch.from_numpy(np.asarray(np.transpose(ns_batch, (0, 3, 1, 2)))).type(dtype) / 255.0)
                nd_mask = Variable(torch.from_numpy(1 - np.asarray(done_mask))).type(dtype)
                # print('Size', s_batch.size())

                # To Be Viewed 
                current_Q_values = self.Q(s_batch).gather(1, a_batch.unsqueeze(1))
                next_max_q = self.target_Q(ns_batch).detach().max(1)[0]
                next_Q_values = nd_mask * next_max_q
                target_Q_values = r_batch + (self.gamma * next_Q_values)
                loss = self.loss_func(current_Q_values, target_Q_values)
                if t % 1000 == 0:
                    print(loss)

                self.opt.zero_grad()
                loss.backward()

                self.opt.step()
                num_updates += 1

                if num_updates % target_update_freq == 0:
                    self.target_Q.load_state_dict(self.Q.state_dict())


                
            if t % 10000 == 0 and t > self.buffer_size:
                self.save_model()
            

            # with open(self.training_record, 'a') as f:
            #     f.write('%d, %d\n' %(num_episode, mean_episode_reward))


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

        if test:
            pass
        else:
            pass

        return self.env.get_random_action()

