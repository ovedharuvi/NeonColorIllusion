import colorsys

from ImageUtils import *


def get_fill_hue(hue):
    r, g, b = [x / 255.0 for x in hue]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - l * 0.15
    rgb_hue = colorsys.hls_to_rgb(h, l, s)
    hue_after = [int(x * 255.0) for x in rgb_hue]
    return hue_after


def diffusion(img, contours):
    white = skimage.img_as_ubyte(np.ones(img.shape))
    for cnt, line_hue in contours:
        if cv2.contourArea(cnt) > np.sqrt(img.shape[0] * img.shape[1]):
            hue = get_fill_hue(line_hue)
            cv2.fillPoly(white, [cnt], color=hue)
    return white
