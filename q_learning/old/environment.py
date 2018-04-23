import numpy as np
from loader import Loader

class Environment:
    def __init__(self, frag=50, train=True):
        ## variables to keep
        self.loader = None
        self.train = train
        self.offset = 0.5 / frag ## states would be frag+5
        self.runtime = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def get_boxes(self, image):
	boxes = []
	for b in image:
            if b[1] >= 0.5:
                boxes.append([b[0], b[2], b[3], b[4], b[5]]) ## class number and x,y coordinate
        return np.array(boxes)

    def get_state(self, action=0): ## state, boxes, answer
        image, answer = self.loader.load()
        image_num = self.loader.total_count
        if not isinstance(image, np.ndarray):
            return None, None, None
        ## calculate state
        image = np.array(sorted(image, key=lambda x: x[1], reverse=True))
        avg = 0.0
        if len(image) >= 5:
            for i in range(5):
                avg += image[i, 1]
            avg /= 5
        else:
            if len(image) == 0:
                avg = 0
            else:
                for a in image:
                    avg += a[1]
                avg /= len(image)
        if avg - 0.5 > 0:
            state = int((avg - 0.5 + (self.offset*5 - np.sum(self.runtime)))/self.offset)
        else:
            state = int((self.offset*5 - np.sum(self.runtime))/self.offset)

        ## for next state
        if action:
            self.runtime[image_num%5] = self.offset
        else:
            self.runtime[image_num%5] = 0.0

        return (state, self.get_boxes(image), answer)

    def intersection_over_union(self, boxA, boxB): ## IoU
        # determine the (x, y)-coordinates of the intersection rectangle
        if (len(boxA) == 0 or len(boxB) == 0):
            return 0
        else:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)

            # return the intersection over union value
            return iou

    def reward(self, boxes, answer, action): ## reward
        total_reward = 0
        for bans in answer:
            highest = 0
            for b in boxes:
                if b[0] == bans[0]:
                    curr = self.intersection_over_union(bans[1:], b[1:])
                    if curr > highest:
                        highest = curr
            total_reward += highest
        total_reward /= len(answer)
        return total_reward if action else total_reward * 5


    def reset(self):
        self.loader = Loader(train=self.train)
        return self.get_state()

    def step(self, state_list, action):
        ## get state
        (state, boxes, answer) = state_list
       ## get reward
        if action:
            boxes = self.get_boxes(self.loader.load_big())
        r = self.reward(boxes, answer, action)

        ## get next state
        next_state_list = self.get_state(action)
        if isinstance(next_state_list[0], int):
            return next_state_list, r, False
        else:
            return None, None, True
