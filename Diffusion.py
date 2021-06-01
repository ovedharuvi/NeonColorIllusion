from skimage.draw import line

from ImageUtils import *


def get_central_point(hull_points):
    x_list = [vertex[0] for vertex in hull_points]
    y_list = [vertex[1] for vertex in hull_points]
    length = len(hull_points)
    x = sum(x_list) / length
    y = sum(y_list) / length
    return np.array([[int(y), int(x)]])


def get_line(point, central_point):
    # being start and end two point (x1,y1), (x2,y2)
    discrete_line = np.array(list(zip(*line(*(point[0][1], point[0][0]), *(central_point[0][1], central_point[0][0])))))

    return discrete_line


def draw_diffusion_line(white_img, line, hue_at_edge, starting_point):
    pass


def diffusion(img, contours):
    white = skimage.img_as_ubyte(np.ones(img.shape))
    for cnt, hue in contours:
        if cv2.contourArea(cnt) > np.sqrt(img.shape[0] * img.shape[1]):
            convex_hull = cv2.convexHull(cnt)
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask, [convex_hull], 0, 255, -1)
            pixel_points = np.transpose(np.nonzero(mask))
            central_point = get_central_point(pixel_points)
            for point in convex_hull:
                line = get_line(point, central_point)
                draw_diffusion_line(white, line, hue, point)
    return white
