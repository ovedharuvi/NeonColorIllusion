import cv2
import imutils
import numpy as np
import scipy
import scipy.ndimage as ndimage
import skimage
import matplotlib.pylab as plt
from cv2 import resize as resize
from scipy.stats import norm

from parameters import *


def to_rad(deg):
    return (deg * np.pi) / 180


def to_degrees(rad):
    return (rad / np.pi) * 180


def apply_img_filter(img, f, mode='correlate'):
    if mode == 'correlate':
        return ndimage.correlate(img, f, mode='nearest').transpose()
    elif mode == 'conv':
        return ndimage.convolve(img, f, mode='nearest')


def fspecial_motion_blur(length, angle):
    # First generate a horizontal line across the middle
    shape = (length, length)
    f = np.zeros(shape)
    f[length // 2 + 1, 1:length] = 1

    # Then rotate to specified angle
    f = imutils.rotate_bound(f, angle)
    f = f / sum(f[:])
    return f


def get_gabor_filter(angle=0, length=81, sig=20, gamma=1, lmd=5, psi=0, is_radian=True):
    # get half size
    d = length // 2

    if (not is_radian):
        # degree -> radian
        theta = to_rad(angle)
    else:
        theta = angle

    # prepare kernel
    gabor = np.zeros((length, length), dtype=np.float32)
    l_norm = np.zeros((length, length))

    # each value
    for y in range(length):
        for x in range(length):
            # distance from center
            px = x - d
            py = y - d

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + gamma ** 2 * _y ** 2) / (2 * sig ** 2)) * np.cos(
                2 * np.pi * _x / lmd + psi)
            l_norm[y][x] = np.cos((2 * np.pi / lmd) * _x)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))
    gabor = resize(gabor, GABOR_KERNEL_SIZE)
    l_norm = resize(l_norm, (length // 5, length // 5))

    return gabor, l_norm


def line_filling_sc(img, theta, fac, c=0.05):
    """theta must be degrees (NOT radiant!)"""
    img_NR = np.power(img, 2) / (np.power(img, 2) + c ** 2)
    motion_blur_ker = fspecial_motion_blur(int(fac), 90 - theta)
    blurred_im = 2 * apply_img_filter(img_NR, motion_blur_ker)
    motion_blur_ker = fspecial_motion_blur(int(np.round(fac / 2)), 90 - theta)
    k = max(fac / 4, 1)
    return k * (blurred_im + apply_img_filter(img_NR, motion_blur_ker)).T, img_NR


# bresenham function is the accepted answer of SO's post https://stackoverflow.com/questions/23930274/list-of-coordinates-between-irregular-points-in-python
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))

    return points


def get_params_for_gaussian(pixel, original_img):
    BOX_SIZE = 5

    px_l, px_c = pixel[1], pixel[0]
    upper_px = original_img.shape[0] - 1 if px_l + BOX_SIZE >= original_img.shape[0] else px_l + BOX_SIZE
    lower_px = 0 if px_l - BOX_SIZE < 0 else px_l - BOX_SIZE
    left_px = 0 if px_c - BOX_SIZE < 0 else px_c - BOX_SIZE
    right_px = original_img.shape[1] - 1 if px_c + BOX_SIZE >= original_img.shape[1] else px_c + BOX_SIZE

    # original_img_hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    px_box = original_img[lower_px:upper_px, left_px:right_px]
    if px_box.shape[0] == 0 or px_box.shape[1] == 0:
        return 0.001, 0.001
    grad_mag = np.amax(px_box) - np.amin(px_box)
    grad_score = 255 - grad_mag
    var = 1.5*np.e ** (-1 * grad_score)
    lambda_ret = var  # FIXME
    return lambda_ret, var


def trigger_guess_by_orientation(img, input_image, theta):
    L, L_norm = get_gabor_filter(to_rad(0))
    img_rotated = rotate_img(img, theta)
    input_image_rotated = rotate_img(input_image, theta)
    conv_norm = L_norm[int(np.ceil(L.shape[0] / 2)), int(np.ceil(L.shape[0] / 2))]
    img_o = apply_img_filter(img_rotated, L, mode='conv') / conv_norm
    img_p = np.maximum(img_o, np.zeros(img_o.shape))
    img_p = normalize(img_p)
    img_p = skimage.img_as_ubyte(normalize(img_p))
    ret, img_p = cv2.threshold(img_p, POST_GABOR_THRESHOLD, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
    dilated_img = cv2.dilate(img_p, kernel, iterations=DILATION_ITERATIONS)
    contours, hier = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    real_contours = []
    pixels_for_guesses = []
    for c in contours:
        if cv2.contourArea(c) > dilated_img.shape[0] * dilated_img.shape[1] / CONTOUR_AREA_THRESH:
            real_contours.append(c)
            topmost = tuple(c[c[:, :, 1].argmin()][0])
            bottommost = tuple(c[c[:, :, 1].argmax()][0])
            pixels_for_guesses.append(topmost)
            pixels_for_guesses.append(bottommost)
    img_orientation_guess = np.zeros(dilated_img.shape)
    for px in pixels_for_guesses:
        lamda, var = get_params_for_gaussian(px, dilated_img)
        img_orientation_guess = trigger_linear_guess(img_orientation_guess, var, px)
        img_orientation_guess = trigger_ort_guess(img_orientation_guess, var, px)

    img_orientation_guess = skimage.img_as_ubyte(normalize(img_orientation_guess))



    # cv2.drawContours(new_img , real_contours , -1 , (255,255,0) , 1)
    print(len(real_contours))
    # cv2.imshow("contours", img_orientation_guess)
    img_orientation_guess = rotate_img(img_orientation_guess, -theta)
    img_n = np.maximum(-img_o, np.zeros(img_o.shape))
    return img_orientation_guess


def strel_line(length, degrees):
    if length >= 1:
        theta = degrees * np.pi / 180
        x = round((length - 1) / 2 * np.cos(theta))
        y = -round((length - 1) / 2 * np.sin(theta))
        points = bresenham(-x, -y, x, y)
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        n_rows = int(2 * max([abs(point_y) for point_y in points_y]) + 1)
        n_columns = int(2 * max([abs(point_x) for point_x in points_x]) + 1)
        strel = np.zeros((n_rows, n_columns))
        rows = ([point_y + max([abs(point_y) for point_y in points_y]) for point_y in points_y])
        columns = ([point_x + max([abs(point_x) for point_x in points_x]) for point_x in points_x])
        idx = []
        for x in zip(rows, columns):
            idx.append(np.ravel_multi_index((int(x[0]), int(x[1])), (n_rows, n_columns)))
        strel.reshape(-1)[idx] = 1

    return skimage.img_as_ubyte(strel).T


def strech(img):
    img_cpy = np.copy(img)
    max_val = np.max(img_cpy)
    min_val = np.maximum([[-1 for i in range(img.shape[1])] for j in range(img.shape[0])], np.min(img_cpy))

    img_cpy = np.maximum(img, min_val)
    img_cpy = img_cpy - min_val
    img_cpy = img_cpy / max_val
    return img_cpy


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def trigger_linear_guess(img_gauss_guess, var, px):
    GAUSS_SIZE = 101
    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), GAUSS_SIZE)
    gauss_line = norm.pdf(x, loc=0, scale=var)
    l, c = px[1], px[0]
    j = 0
    for i in range(-GAUSS_SIZE // 2, GAUSS_SIZE // 2):
        if l+i >= img_gauss_guess.shape[0]:
            break
        img_gauss_guess[l+i][c] = gauss_line[j]
        j += 1
    return img_gauss_guess


def trigger_ort_guess(img_gauss_guess, var, px):
    GAUSS_SIZE = 101
    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), GAUSS_SIZE)
    gauss_line = norm.pdf(x, loc=0, scale=var)
    l, c = px[1], px[0]
    j = 0
    for i in range(-GAUSS_SIZE//2, GAUSS_SIZE//2):
        if c+i >= img_gauss_guess.shape[1]:
            break
        img_gauss_guess[l][c + i] = gauss_line[j]
        j += 1
    return img_gauss_guess



def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def rotate_img(img, theta):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
