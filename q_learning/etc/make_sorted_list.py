import os

alist = os.listdir('/SSD/scam/answers/')
alist.sort()

f = open('sorted_list.txt', 'w')

for a in alist:
    f.write(a.replace('.xml','')+'\n')
