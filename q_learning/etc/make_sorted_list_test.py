import os

alist = os.listdir('/SSD/scam/test_answers/')
alist.sort()

f = open('sorted_test.txt', 'w')

for a in alist:
    f.write(a.replace('.xml','')+'\n')
