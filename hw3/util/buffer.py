import random

class Buffer(object):
    def __init__(self, size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.size = size

    def append(self, state, action, reward, next_state, done):

        if len(self.states) > self.size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size=32):
        random_list = random.sample(range(len(self.states)-1), batch_size)
        
        return_states = []
        return_actions = []
        return_rewards = []
        return_next_states = []
        return_dones = []

        for i in random_list:
            return_states.append(self.states[i])
            return_actions.append(self.actions[i])
            return_rewards.append(self.rewards[i])
            return_next_states.append(self.next_states[i])
            return_dones.append(self.dones[i])

        return return_states, return_actions, return_rewards, return_next_states, return_dones