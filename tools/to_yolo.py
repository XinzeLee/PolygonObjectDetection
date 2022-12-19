import os
import cv2
import json
import numpy as np


def convert():
    image_root = "/Users/qianwang/datasets/ccpd/images"
    json_name = "/Users/qianwang/datasets/ccpd/images/ccpd_green_val.json"
    with open(json_name, "r") as f:
        anns = json.load(f)
    for k in anns:
        image_name = os.path.join(image_root, k)
        # image = cv2.imread(image_name)
        # print(image.shape)
        save_name = os.path.join("../", "ccpd", "labels", os.path.basename(k).replace('.jpg', '.txt'))
        txt = open(save_name, "w")
        line = "0"
        ann = anns[k]
        keypoints = ann['poly'][0]
        for i in range(0, 8, 2):
            x = np.round(keypoints[i] / 720, 6)
            y = np.round(keypoints[i+1] / 1160, 6)
            line += " "
            line += str(x)
            line += " "
            line += str(y)
        line += "\n"
        txt.write(line)


convert()