from xml.etree.ElementTree import parse
import numpy as np

class XMLparser:
    def __init__(self):
        self.class_dict = {'sheep': 16, 'horse': 12, 'bicycle': 1, 'aeroplane': 0, 'cow': 9, 'sofa': 17, 'bus': 5, 'dog': 11, 'cat': 7, 'person': 14, 'train': 18, 'diningtable': 10, 'bottle': 4, 'car': 6, 'pottedplant': 15, 'tvmonitor': 19, 'chair': 8, 'bird': 2, 'boat': 3, 'motorbike': 13}

    def parsing(self, name):
        tree = parse(name)
        note = tree.getroot()
        objects = note.findall("object")
        data = []
        for e in objects:
            c = e.findtext("name")
            bbox = e.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            data.append([self.class_dict[c], xmin, ymin, xmax, ymax])
        return np.array(data)
