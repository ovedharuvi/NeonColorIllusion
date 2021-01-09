import numpy as np
import imutils
import scipy.ndimage as ndimage


def apply_img_filter(img, f, mode='correlate'):
    if mode == 'correlate':
        return ndimage.correlate(img, f, mode='nearest').transpose()
    elif mode == 'conv':
        return ndimage.convolve(img, f, mode='nearest')


def fspecial_motion_blur(length, angle):
    # First generate a horizontal line across the middle
    f = np.zeros(length)
    f[np.floor(length / 2) + 1, 1:length] = 1

    # Then rotate to specified angle
    f = imutils.rotate_bound(f, angle)
    f = f / sum(f[:])
    return f


def line_filling_sc(img, theta, fac, c=0.05):
    img_NR = np.power(img, 2) / (np.power(img, 2) + c ^ 2)
    motion_blur_ker = fspecial_motion_blur(fac, 90-theta)
    blurred_im = 2 * apply_img_filter(img_NR, motion_blur_ker)
    motion_blur_ker = fspecial_motion_blur(np.round(fac/2), 90-theta)
    k = max(fac/4, 1)
    return k * (blurred_im + apply_img_filter(img_NR, motion_blur_ker))
