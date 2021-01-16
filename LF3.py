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


def line_filling(input_image):
    LDR = 0
    img = preprocess(input_image)
    scale_levels = np.arange(1,5)
    im_scel = np.zeros(img.shape)
    im_scel_tot = np.zeros(img.shape)
    thetas = np.arange(0, 360, 30)
    kk = len(thetas)
    tic = time.perf_counter()
    for scale in scale_levels:
        rs_factor_rows = round(img.shape[0] / max(1, 2 * (scale - 1)))
        rs_factor_colomns = round(img.shape[1] / max(1, 2 * (scale - 1)))
        scaled_img = cv2.resize(img, (rs_factor_rows, rs_factor_colomns))
        imm_neg_or_lin = np.zeros(scaled_img.shape)
        imm_pos_or_lin = np.zeros(scaled_img.shape)
        imm_neg_or_ort = np.zeros(scaled_img.shape)
        imm_pos_or_ort = np.zeros(scaled_img.shape)
        for theta in thetas:
            edges_p, edges_n = detect_edges(scaled_img, theta)
            structured_element = strel_line(10, theta - 90)
            img_n_dilated = cv2.dilate(edges_n, kernel=structured_element)
            img_p_dilated = cv2.dilate(edges_p, kernel=structured_element)
            factor = max(5, 25. / scale)
            line_filled_img_neg, neg_img_nr = line_filling_sc(img_n_dilated, theta, factor)
            line_filled_img_pos, pos_img_nr = line_filling_sc(img_p_dilated, theta, factor)
            line_filled_img_neg = 0.5 * np.maximum(np.zeros(line_filled_img_neg.shape), line_filled_img_neg)
            line_filled_img_pos = 0.5 * np.maximum(np.zeros(line_filled_img_pos.shape), line_filled_img_pos)
            imm_neg_or_lin += line_filled_img_neg
            imm_pos_or_lin += line_filled_img_pos

            line_filled_img_neg, neg_img_nr = line_filling_sc(img_n_dilated, theta - 90, factor)
            line_filled_img_pos, pos_img_nr = line_filling_sc(img_p_dilated, theta - 90, factor)
            line_filled_img_neg = 0.5 * np.maximum(np.zeros(line_filled_img_neg.shape), line_filled_img_neg - neg_img_nr)
            line_filled_img_pos = 0.5 * np.maximum(np.zeros(line_filled_img_neg.shape), line_filled_img_pos - pos_img_nr)
            imm_neg_or_ort += line_filled_img_neg
            imm_pos_or_ort += line_filled_img_pos

        IM2scel = -imm_neg_or_lin + imm_pos_or_lin
        im_scel = cv2.resize(IM2scel.T, img.shape)
        im_scel = im_scel / (2. ** (scale - 1))
        im_scel_tot = im_scel_tot + im_scel.T

    toc = time.perf_counter()
    streched_img = strech(im_scel_tot)
    return streched_img
