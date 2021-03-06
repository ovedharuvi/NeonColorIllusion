import numpy as np
import cv2
import matplotlib.pylab as plt
import skimage
from skimage import color as color
import time
from LFutils import *


def preprocess(input_image):
    img = skimage.img_as_float(input_image)
    img_dim = img.shape[2]
    if img_dim > 1:
        img_hsv = color.rgb2hsv(img)
        return img_hsv[:, :, 2]
    else:
        return img


def detect_false_contour(input_image):
    LDR = 0
    img = preprocess(input_image)
    guesses_sum = np.zeros(img.shape)
    thetas = np.arange(0, 360, 30)
    kk = len(thetas)
    tic = time.perf_counter()
    for theta in thetas:
        guesses_sum += trigger_guess_by_orientation(img, theta)

    x=1
    return guesses_sum


