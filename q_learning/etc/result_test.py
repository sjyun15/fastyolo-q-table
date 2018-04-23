import numpy as np

action_list = np.load('action_list.npy')
big_images = np.where(action_list == 1.0)
print len(big_images[0])
