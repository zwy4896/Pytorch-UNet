import argparse
import base64
import json
import os
import PIL.Image
import cv2
import numpy as np


def main(img_dir):
    for root, dir, files in os.walk(img_dir):
        for file in files:
            if '.json' not in file:
                continue
            data = load_json(file)
            shapes = data['shapes']
            img_h = data['imageHeight']
            img_w = data['imageWidth']
            points = []
            for shape in shapes:
                point = shape['points']
                points.append(point)
            img = np.zeros((img_h, img_w))
            for p in points:
                p = np.array(p, dtype=int)
                cv2.fillPoly(img, [p], 255)
                cv2.imwrite(os.path.join('/home/trd/code/Pytorch-UNet/data/masks', file.split('.')[0]+'.jpg'), img)
                # cv2.imshow('test', img)
                # cv2.waitKey(0)

def load_json(json_file):
    with open(os.path.join(img_dir, json_file), 'r') as f:
        data = json.load(f)

    return data
if __name__ == "__main__":
    img_dir = '/home/trd/share/DATASET/train'
    main(img_dir)