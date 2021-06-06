# This is a sample Python script.

import logging

from skimage import io

from Diffusion import diffusion
from FalseContours import detect_false_contour
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ImageUtils import *
from Params import *

logging.basicConfig(level=logging.INFO)
img = io.imread(image_path)
img = init_img(img)
orig = img
logging.info('Image shape:{}, Image data type:{}'.format(img.shape, img.dtype))
# plt.figure(figsize=(30,15))
if SAVE:
    filename = 'Results/InputImage.jpg'
    skimage.io.imsave(filename, img)

img = preprocess(img)
guesses_sum, threshold_guesses_sum = detect_false_contour(img, orig)
guesses_sum = post_process(guesses_sum)
if SAVE:
    filename = f'Results/guesses_aggregater_before_threshold.jpg'
    skimage.io.imsave(filename, guesses_sum)
threshold_guesses_sum = post_process(threshold_guesses_sum)
if SAVE:
    filename = f'Results/guesses_aggregated_after_threshold.jpg'
    skimage.io.imsave(filename, threshold_guesses_sum)
#
# plt.subplot(132)
# plt.imshow(threshold_guesses_sum, cmap='gray')
# plt.title("Educated guesses with threshold")
# plt.xticks([]), plt.yticks([])
#
# plt.imshow(guesses_sum, cmap='gray')
# plt.title("Guesses without threshold")
# plt.xticks([]), plt.yticks([])  # to hide tick values on X & Y axis
# plt.show()

# if orig.shape[2] >= 3:
#     threshold_guesses_sum = skimage.img_as_ubyte(skimage.color.gray2rgb(threshold_guesses_sum))
orig_final = skimage.img_as_ubyte(orig)
line_completion, contours = draw_contours(orig_final, threshold_guesses_sum, IS_GRAY)
if SAVE:
    filename = 'Results/finalLineCompletion.jpg'
    skimage.io.imsave(filename, line_completion)

diffusion = diffusion(orig_final, contours)
dst = cv2.addWeighted(orig_final, 0.5, diffusion, 0.5, 0)

if SAVE:
    filename = 'Results/diffusionOnly.jpg'
    skimage.io.imsave(filename, diffusion)
    filename = 'Results/finalDiffusion.jpg'
    skimage.io.imsave(filename, dst)

# plt.subplot(133)
# plt.title("Line filling algorithm result")
# plt.xticks([]), plt.yticks([])
# plt.imshow(final, cmap='gray')
# plt.show()
