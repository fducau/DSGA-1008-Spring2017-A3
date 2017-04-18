import cv2
import os
import sys
import numpy as np


bg_folder = './data/scenic/'
fg_folder = './data/animal_database/'
N = 10000   # Dataset size


def get_image_path(root, img_type='jpg'):
    walks = []
    paths = []

    for w in os.walk(root):
        walks.append(w)

    for w in walks:
        w_path = [w[0] + '/' + i for i in w[2] if i[-3:] == img_type]
        paths = paths + w_path

    return paths



def main():
    fg_img_files = get_image_path(fg_folder, 'jpg')
    # fg_segmentation_files = get_image_path(fg_folder, 'png')
    bg_files = get_image_path(bg_folder)

    fg_img_files = np.random.choice(fg_img_files, size=N, replace=True)
    bg_segmentation_files = np.array([p.replace('original', 'segment').replace('jpg', 'png') for p in fg_img_files])
    bg_files = np.random.choice(bg_files, size=N, replace=True)

if __name__ == '__main__':
    main()