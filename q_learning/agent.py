import numpy as np
import random
import time
from environment import Environment

class Agent:
    def __init__(self, frag=50, train=True):
        ## action: 0(little)/1(big)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
	## for epsilon-greedy
        if train:
            self.eps_max = 0.99
        else:
            self.eps_max = 0.1
        self.eps_min = 0.1
        self.eps_diff = 5e-7
        if train:
            self.q_table = np.zeros(shape=(frag, 5, 2))
        else:
            self.q_table = np.load('q_table.npy')
        self.action_list = []
        self.train = train

    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    def random_action(self):
        return random.randint(0,1)

    def select_action(self, state):
        # Select an action according e-greedy. You need to use a random-number generating function and add a library if necessary.
        if self.train:
            e = random.uniform(0.0,1.0)
            if self.eps_max > self.eps_min:
                self.eps_max -= self.eps_diff
            if e <= self.eps_max:
                return self.random_action()
            else:
                return np.argmax(self.q_table[state])
        else:
            return np.argmax(self.q_table[state])
