import xml.etree.ElementTree as ET
import pickle
import os
import numpy as np
from os import listdir, getcwd
from os.path import join
from itertools import combinations

with open("/NAS/jeonghyun/esweek/class_dic.pkl", "rb") as handle:
    classes = pickle.load(handle)

imageset = open("/SSD/scam/train_annots.txt")

score_table = np.zeros(shape=(30), dtype=int)

for i, s in enumerate(imageset):
    if i%10000 == 0:
        print("{} images processed...".format(i))
    name = s[:-1]
    in_file = open(name)
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        else:
            score_table[classes[cls]] += 1

imageset.close()

f = open("class_count.csv", "w")

for idx, s in enumerate(score_table):
    f.write("{},{}\n".format(idx, s))
