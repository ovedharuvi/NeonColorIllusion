# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from LF3 import *
from LFutils import *
import numpy as np
import cv2
import matplotlib.pylab as plt
from skimage import transform, io , color

img = io.imread('neon_7.gif')
img = img[:, :, :-1]
print('Image shape:{}, Image data type:{}'.format(img.shape, img.dtype))
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X & Y axis
plt.show()

filled_img = line_filling(img)
plt.imshow(filled_img, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X & Y axis
plt.show()

img = color.rgb2gray(img)
plt.imshow(img - filled_img, cmap='gray')
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
