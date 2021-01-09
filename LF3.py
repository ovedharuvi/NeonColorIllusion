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
        return img_hsv[:, :, 3]
    else:
        return img


def line_filling(input_image):
    LDR = 1
    img = preprocess(input_image)
    scale_levels = 4
    im_scel = np.zeros((img.shape, scale_levels))
    im_scel_tot = np.zeros(img.shape)
    thetas = np.arange(0, 360, 30)
    kk = len(thetas)
    tic = time.perf_counter()
    for i in range(scale_levels):
        rs_factor_rows = round(img.shape[0] / max(1, 2 * (i - 1)))
        rs_factor_colomns = round(img.shape[1] / max(1, 2 * (i - 1)))
        scaled_img = cv2.resize(img, (rs_factor_rows, rs_factor_colomns))
        for j in range(2, kk-1):
            tat = np.pi*(j-1)/kk
            L , L_norm = get_gabor_filter(tat)
            conv_norm = L_norm(np.ceil(L.shape[0]/2), np.ceil(L.shape[0]/2))
            img_o = apply_img_filter(scaled_img, L , mode= 'conv')
            img_p = max(img_o , 0)
            img_n = max(-img_o , 0)






def get_gabor_filter(theta):
    pass