import numpy as np
import sys
import os

class_dict = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

result_prefix = "/SSD/scam/q_learning/output/comp4_det_test_"
if sys.argv[1] == 'test':
    big_prefix = "/SSD/scam/test_big/"
    little_prefix = "/SSD/scam/test_little/"
else:
    big_prefix = "/SSD/scam/big/"
    little_prefix = "/SSD/scam/little/"

actions = np.load('action_list.npy')

blist = os.listdir(big_prefix)
llist = os.listdir(little_prefix)

blist.sort()
llist.sort()

total_num = len(blist)
big_count = 0

for idx in range(total_num):
    if idx % 100 == 0:
        print("{} images done".format(idx))
    name = blist[idx].replace(".npy", "")
    if actions[idx]:
        big_count += 1
        curr_boxes = np.load(big_prefix+blist[idx])
    else:
        curr_boxes = np.load(little_prefix+llist[idx])

    for bidx, box in enumerate(curr_boxes):
        curr_class = class_dict[int(box[0])]
        if bidx == 0:
            pre_class = curr_class
            fresult = open(result_prefix+curr_class+".txt", "a")
        elif pre_class != curr_class:
            fresult = open(result_prefix+curr_class+".txt", "a")
        fresult.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(name, box[1], box[2], box[3], box[4], box[5]))

print ("big ratio: {}/{} = {:.4f}".format(big_count, total_num, float(big_count)/float(total_num)))
