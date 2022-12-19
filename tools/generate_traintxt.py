import os

def gen():
    root = "../ccpd/images/"
    paths = os.listdir(root)
    txts = open("../ccpd/train.txt", "w")
    val_txts = open("../ccpd/val.txt", "w")
    for p in paths:
        name = os.path.join("../ccpd/images", p)
        line = name + "\n"
        txts.write(line)
        val_txts.write(line)


gen()
