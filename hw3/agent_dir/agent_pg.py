from agent_dir.agent import Agent
import scipy
import numpy as np

import torch.optim as optim

def prepro(o,image_size=[80,80]):
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


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)
        self.policy_network = PolicyNetwork()
        if os.path.isfile('pg_params.pkl'):
            print('Load Policy Network parametets ...')
            self.policy_network.load_state_dict(torch.load('pg_params.pkl'))
        self.opt = optim.RMSprop(self.policy_network.parameters(), lr=1e-4, weight_decay=0.99)
        self.gamma = 0.99

        if args.test_pg:
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

    def finish_episode():
        R = 0
        policy_loss = []
        rewards = []
        for r in policy.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        # turn rewards to pytorch tensor and standardize
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        for log_prob, reward in zip(self.policy_network.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.opt.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.opt.step()

        # clean rewards and saved_actions
        del policy.rewards[:]
        del policy.saved_log_probs[:]

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        running_reward = None
        reward_sum = 0
        for i_episode in count(1):
            state = self.env.reset()
            for t in range(10000):
                action = make_action(state)
                action = action + 1
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                
                self.policy_network.rewards.append(reward)
                if done:
                    # tracking log
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                    reward_sum = 0
                    break
                    
                if reward != 0:
                    print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!'))

            # use policy gradient update model weights
            if i_episode % args.batch_size == 0:
                print('ep %d: policy network parameters updating...' % (i_episode))
                self.finish_episode()

            # Save model in every 50 episode
            if i_episode % 50 == 0:
                print('ep %d: model saving...' % (i_episode))
                torch.save(policy.state_dict(), 'pg_params.pkl')


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
        ##################
        # YOUR CODE HERE #
        ##################
        state = prepro(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.data[0]

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