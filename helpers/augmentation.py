# Script for data augmentation:
import numpy as np
import cv2
import os

# Horizontal flip
def horizontal_flip(image):
    image = cv2.flip(image, 1)
    return image

# Vertical flip
def vertical_flip(image):
    image = cv2.flip(image, 0)
    return image

# Image Rotation
def rotate_image(image):
    image_1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image_2 = cv2.rotate(image, cv2.ROTATE_180)
    image_3 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_1, image_2, image_3

directories = ['A', 'B', 'OUT']
for directory in directories:
    for file in os.listdir(f'../dataset - Copy/train/{directory}'):
        image = cv2.imread(f'../dataset - Copy/train/{directory}/{file}')
        horizontal = horizontal_flip(image)
        vertical = vertical_flip(image)
        rotate_90, rotate_180, rotate_270 = rotate_image(image)

        cv2.imwrite(f'../dataset - Copy/train/{directory}/{file.split(".")[0]}_horizontal.png', horizontal)
        cv2.imwrite(f'../dataset - Copy/train/{directory}/{file.split(".")[0]}_vertical.png', vertical)
        cv2.imwrite(f'../dataset - Copy/train/{directory}/{file.split(".")[0]}_rotate_90.png', rotate_90)
        cv2.imwrite(f'../dataset - Copy/train/{directory}/{file.split(".")[0]}_rotate_180.png', rotate_180)
        cv2.imwrite(f'../dataset - Copy/train/{directory}/{file.split(".")[0]}_rotate_270.png', rotate_270)

    print("Directory done")

