## get xml answer by image number(sorted)

import sys
import os

from xmlparser import XMLparser

a = XMLparser()

if sys.argv[1] == 'train':
    prefix = '/SSD/scam/answers/'
else:
    prefix = '/SSD/scam/test_answers/'

f = open('sorted_list.txt', 'r')
fdict = {}


for i, ff in enumerate(f):
    fdict[i] = ff.replace('\n', '')

name = fdict[int(sys.argv[1])]

print a.parsing(prefix+name+'.xml')
