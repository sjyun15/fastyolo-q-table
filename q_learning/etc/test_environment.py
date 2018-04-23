import numpy as np
import os

from environment import Environment

env = Environment()

a = env.reset()
next_state_list, r, done = env.step(a, 1)

print next_state_list
print r
print done
