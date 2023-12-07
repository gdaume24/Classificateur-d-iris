import cv2
import numpy as np

def preprocess_img(img, new_dim = (240, 320)):

    new_img = cv2.resize(img, (new_dim[1], new_dim[0]), interpolation = cv2.INTER_AREA)
    mean = np.mean(new_img)
    std = np.std(new_img)
    new_img = (new_img - mean) / std

    return new_img

def preprocess_img_without_norm(img, new_dim = (240, 320)):

    new_img = cv2.resize(img, (new_dim[1], new_dim[0]), interpolation = cv2.INTER_AREA)

    return new_img