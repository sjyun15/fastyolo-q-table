import os
import numpy as np
from xmlparser import XMLparser

class Loader:
    def __init__(self, max_num=500, train=True):
        self.max_num = max_num
        self.total_count = -1
        self.curr_count = -1

        if train:
            self.total_image = 16551
        else:
            self.total_image = 4952

        self.big = None ## result of big
        self.little = None ## result of little
        self.answers = None

        if not train:
            self.big_prefix = "/SSD/scam/test_big/"
            self.little_prefix = "/SSD/scam/test_little/"
            self.answer_prefix = "/SSD/scam/test_answers/"
        else:
            self.big_prefix = "/SSD/scam/big/"
            self.little_prefix = "/SSD/scam/little/"
            self.answer_prefix = "/SSD/scam/answers/"

        self.blist = os.listdir(self.big_prefix)
        self.llist = os.listdir(self.little_prefix)
        self.alist = os.listdir(self.answer_prefix)

        self.blist.sort()
        self.llist.sort()
        self.alist.sort()

        self.parser = XMLparser()

    def get_npys(self):
        bound = self.total_count+self.max_num
        if bound >= self.total_image:
            bound = self.total_image
        loading_big = self.blist[self.total_count:bound]
        loading_little = self.llist[self.total_count:bound]
        loading_answers = self.alist[self.total_count:bound]

        self.big = [np.load(self.big_prefix+f) for f in loading_big]
        self.little = [np.load(self.little_prefix+f) for f in loading_little]
        data = []

        for name in loading_answers:
            data.append(self.parser.parsing(self.answer_prefix+name))
        self.answers = data

    def load(self): ## get 1 image
        self.curr_count += 1
        self.total_count += 1

        if self.total_count >= self.total_image: ## for edge case
            return None, None
        if self.curr_count >= self.max_num or self.total_count == 0:
            ## load max_num(default 500) .npy for each(big/little) and get answers
            self.get_npys()
            self.curr_count = 0
        return self.little[self.curr_count], self.answers[self.curr_count]

    def load_big(self): ## always called after calling 'load'
        return self.big[self.curr_count]
