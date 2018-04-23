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

score_table = np.zeros(shape=(30, 30), dtype=int)

for i, s in enumerate(imageset):
	if i%10000 == 0:
		print("{} images processed...".format(i))
	name = s[:-1]
	in_file = open(name)
	tree=ET.parse(in_file)
	root = tree.getroot()
	size = root.find('size')
	w = int(size.find('width').text)
	h = int(size.find('height').text)
	class_of_this_image = []
	for obj in root.iter('object'):
		cls = obj.find('name').text
		if cls not in classes:
			continue
		cls_id = classes[cls]
		class_of_this_image.append(cls_id)
	for idx in combinations(sorted(list(set(class_of_this_image))), 2):
		score_table[idx] += 1

np.save('class_result', score_table)
