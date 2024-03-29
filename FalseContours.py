import logging

import matplotlib.pyplot as plt
import skimage.io
from cv2 import resize
from scipy.stats import norm

from ImageUtils import *
from Params import *

"""Algorithm Explained:
1. For each theta aggregate the guesses by the current theta using gabor filter
2. Filter out weak guesses using threshold.
"""
def detect_false_contour(img, original_img):
    logging.info("Starting detect false contours")
    guesses_sum = np.zeros(img.shape)
    thetas = np.arange(0, 360, THETAS_STEP)
    kk = len(thetas)
    for theta in thetas:
        guesses_sum += trigger_guess_by_orientation(img, original_img, theta) // kk
    nor_filled_img = skimage.img_as_ubyte(normalize(guesses_sum))
    ret, threshold_guesses_sum = cv2.threshold(nor_filled_img, GUESSES_THRESHOLD, 255, cv2.THRESH_TOZERO)
    return guesses_sum, threshold_guesses_sum


def get_gabor_filter(angle=0, length=81, sig=20, gamma=1, lmd=5, psi=0, is_radian=True):
    # get half size
    d = length // 2

    if not is_radian:
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


# bresenham function is the accepted answer of SO's post
#  https://stackoverflow.com/questions/23930274/list-of-coordinates-between-irregular-points-in-python
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
    BOX_SIZE = GAUSSIAN_PARAMS_BOX_SIZE
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
    var = 1.5 * np.e ** (-1 * grad_score)
    lambda_ret = var  # FIXME
    return lambda_ret, var


"""
Algorithm Explained:
1. Rotate the image by theta.
2. Find gradients with Gabor filter.
3. Dilation in order to have continuous lines
4. Filter out "small" lines (by area - reduce noise )
5.  Find the bottom and up most points of the contours.
6. Trigger gaussian guesses (linear and orthogonal) at the found points.
7. Rotate image back to the original orientation.

"""
def trigger_guess_by_orientation(img, orig_image, theta):
    logging.info('Started trigger guess for theta = {}'.format(theta))
    L, L_norm = get_gabor_filter(0)
    if SAVE :
        filename= f'Results/gabor_filter_orientation_0.jpg'
        skimage.io.imsave(filename, L)
    img_rotated = rotate_img(img, theta)
    conv_norm = L_norm[int(np.ceil(L.shape[0] / 2)), int(np.ceil(L.shape[0] / 2))]
    img_o = apply_img_filter(img_rotated, L, mode='conv') / conv_norm
    img_p = np.maximum(img_o, np.zeros(img_o.shape))
    img_p = normalize(img_p)
    img_p = skimage.img_as_ubyte(normalize(img_p))
    if SAVE:
        if theta == 0:
            filename = f'Results/OrientationDetection_theta_{theta}.jpg'
            skimage.io.imsave(filename, img_p)
    kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
    dilated_img = cv2.dilate(img_p, kernel, iterations=DILATION_ITERATIONS)
    ret, dilated_img = cv2.threshold(dilated_img, POST_GABOR_THRESHOLD, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    logging.info('For theta = {} , Number of contours found = {}'.format(theta, len(contours)))
    real_contours = []
    pixels_for_guesses = []
    for c in contours:
        if cv2.contourArea(c) > np.sqrt(orig_image.shape[0] * orig_image.shape[1]):
            real_contours.append(c)
            topmost = tuple(c[c[:, :, 1].argmin()][0])
            bottommost = tuple(c[c[:, :, 1].argmax()][0])
            pixels_for_guesses.append(topmost)
            pixels_for_guesses.append(bottommost)
    img_orientation_guess = np.zeros(dilated_img.shape)
    if len(contours) == 0:
        return img_orientation_guess
    for px in pixels_for_guesses:
        lamda, var = get_params_for_gaussian(px, dilated_img)
        img_orientation_guess = trigger_linear_guess(img_orientation_guess, var, px)
        img_orientation_guess = trigger_ort_guess(img_orientation_guess, var, px)
    img_orientation_guess = rotate_img(img_orientation_guess, -theta)
    img_orientation_guess = skimage.img_as_ubyte(normalize(img_orientation_guess))
    if SAVE:
        filename = f'Results/orientation_educated_guess_theta_{theta}.jpg'
        skimage.io.imsave(filename, img_orientation_guess)
    return img_orientation_guess


def strel_line(length, degrees):
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


def trigger_linear_guess(img_gauss_guess, var, px):
    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), GAUSS_SIZE)
    gauss_line = norm.pdf(x, loc=0, scale=var)
    l, c = px[1], px[0]
    j = 0
    for i in range(-GAUSS_SIZE // 2, GAUSS_SIZE // 2):
        if l + i >= img_gauss_guess.shape[0]:
            break
        img_gauss_guess[l + i][c] = gauss_line[j]
        j += 1
    return img_gauss_guess


def trigger_ort_guess(img_gauss_guess, var, px):
    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), GAUSS_SIZE)
    gauss_line = norm.pdf(x, loc=0, scale=var)
    l, c = px[1], px[0]
    j = 0
    for i in range(-GAUSS_SIZE // 2, GAUSS_SIZE // 2):
        if c + i >= img_gauss_guess.shape[1]:
            break
        img_gauss_guess[l][c + i] = gauss_line[j]
        j += 1
    return img_gauss_guess


def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()
