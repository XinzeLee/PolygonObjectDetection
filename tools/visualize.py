import os
import cv2


def visual():
    # image_path = "../UCAS50/images/train/"
    image_path = "../ccpd/images/"
    paths = os.listdir(image_path)
    for p in paths:
        image_name = os.path.join(image_path, p)
        image = cv2.imread(image_name)
        h, w, c = image.shape
        label_name = image_name.replace("images", "labels").replace(".jpg", ".txt")
        txts = open(label_name, "r").readlines()
        for line in txts:
            line = line.strip("\n").split(" ")
            line = line[1:]
            points = []
            for i in range(0, 8, 2):
                x = int(float(line[i]) * w)
                y = int(float(line[i+1]) * h)
                points.append(x)
                points.append(y)
            for i in range(0, 6, 2):
                cv2.line(image, (points[i], points[i+1]), (points[i+2], points[i+3]), (255, 0, 0), 2)
        cv2.imshow("i", image)
        cv2.waitKey(0)


visual()