import numpy as np
import imutils
import scipy.ndimage as ndimage


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
    f = np.zeros(length)
    f[np.floor(length / 2) + 1, 1:length] = 1

    # Then rotate to specified angle
    f = imutils.rotate_bound(f, angle)
    f = f / sum(f[:])
    return f


def get_gabor_filter(theta, lmd=12, sig=8, x_b=13, y_b=13, length=25):
    """theta must be radiant (NOT degrees!)"""
    xs = ys = [k for k in range(length)]
    l = np.zeros((length, length))
    l_norm = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            exp = np.exp(np.e, -((xs[i]-x_b) ^ 2 + (ys[j]-y_b) ^ 2)/(sig ^ 2))
            k = (xs[i]-x_b) * np.cos(theta) + (ys[i]-y_b) * np.sin(theta)
            l[j][i] = np.cos((2*np.pi / lmd) * k)
            l_norm[j][i] = l[j][i]
            l[j][i] *= exp
    return l, l_norm


def line_filling_sc(img, theta, fac, c=0.05):
    """theta must be degrees (NOT radiant!)"""
    img_NR = np.power(img, 2) / (np.power(img, 2) + c ^ 2)
    motion_blur_ker = fspecial_motion_blur(fac, 90-theta)
    blurred_im = 2 * apply_img_filter(img_NR, motion_blur_ker)
    motion_blur_ker = fspecial_motion_blur(np.round(fac/2), 90-theta)
    k = max(fac/4, 1)
    return k * (blurred_im + apply_img_filter(img_NR, motion_blur_ker))
