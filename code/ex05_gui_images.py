import cv2
import argparse
import numpy as np
from utils import images
from typing import Optional


TITLE_WINDOW = 'Example5'
RANGE = 10
IMG: Optional[np.ndarray] = None
M = np.array([[1, 0], [0, 1]])
T = np.array([[0, 0]]).T
C = np.array([[0, 0]]).T


def create_on_trackbar_set_M(x, y):
    def on_trackbar(val):
        global TITLE_WINDOW, IMG, M, T, C
        M[x, y] = val
        img = images.fast_transform(IMG.copy(), M=M, T=T, trans_point=C)

        cv2.imshow(TITLE_WINDOW, img)

    return on_trackbar


def create_on_trackbar_set_T(x):
    def on_trackbar(val):
        global TITLE_WINDOW, IMG, M, T, C
        T[x, 0] = val
        img = images.fast_transform(IMG.copy(), M=M, T=T, trans_point=C)

        cv2.imshow(TITLE_WINDOW, img)

    return on_trackbar


def create_on_trackbar_set_C(x):
    def on_trackbar(val):
        global TITLE_WINDOW, IMG, M, T, C
        C[x, 0] = val
        img = images.fast_transform(IMG.copy(), M=M, T=T, trans_point=C)

        cv2.imshow(TITLE_WINDOW, img)

    return on_trackbar


def main(args):
    global IMG
    IMG = images.load_image(args.input)

    cv2.namedWindow(TITLE_WINDOW)

    # Create trackbar functions
    trackbar_m11 = create_on_trackbar_set_M(0, 0)
    trackbar_m12 = create_on_trackbar_set_M(0, 1)
    trackbar_m21 = create_on_trackbar_set_M(1, 0)
    trackbar_m22 = create_on_trackbar_set_M(1, 1)
    trackbar_t1 = create_on_trackbar_set_T(0)
    trackbar_t2 = create_on_trackbar_set_T(1)
    trackbar_c1 = create_on_trackbar_set_C(0)
    trackbar_c2 = create_on_trackbar_set_C(1)

    # Create trackbars
    global RANGE
    cv2.createTrackbar('M11', TITLE_WINDOW, 0, RANGE, trackbar_m11)
    cv2.createTrackbar('M12', TITLE_WINDOW, 0, RANGE, trackbar_m12)
    cv2.createTrackbar('M21', TITLE_WINDOW, 0, RANGE, trackbar_m21)
    cv2.createTrackbar('M22', TITLE_WINDOW, 0, RANGE, trackbar_m22)
    cv2.createTrackbar('T1', TITLE_WINDOW, 0, IMG.shape[1], trackbar_t1)
    cv2.createTrackbar('T2', TITLE_WINDOW, 0, IMG.shape[0], trackbar_t2)
    cv2.createTrackbar('C1', TITLE_WINDOW, 0, IMG.shape[1], trackbar_c1)
    cv2.createTrackbar('C2', TITLE_WINDOW, 0, IMG.shape[0], trackbar_c2)

    # Set tracker min values
    cv2.setTrackbarMin('M11', TITLE_WINDOW, -RANGE)
    cv2.setTrackbarMin('M12', TITLE_WINDOW, -RANGE)
    cv2.setTrackbarMin('M21', TITLE_WINDOW, -RANGE)
    cv2.setTrackbarMin('M22', TITLE_WINDOW, -RANGE)

    # Show some stuff
    trackbar_c1(IMG.shape[1] // 2)
    trackbar_c2(IMG.shape[0] // 2)
    # Wait until user press some key
    cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to the first input image.', default='first.jpg')
    pargs = parser.parse_args()
    main(pargs)
