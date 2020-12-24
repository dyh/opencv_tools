import cv2 as cv
import numpy as np

from utils import plt_show


def show_outline(src):
    contours = cv.Canny(image=src, threshold1=125, threshold2=350)
    plt_show(255 - contours, title='Canny Contours')

    image_gray = cv.cvtColor(src=src, code=cv.COLOR_BGR2GRAY)

    contours = cv.Canny(image=image_gray, threshold1=125, threshold2=350)
    plt_show(255 - contours, title='Canny Contours Gray')

    # Hough tranform for line detection
    theta = np.pi / 180
    threshold = 50
    lines = cv.HoughLinesP(image=contours, rho=1, theta=theta, threshold=threshold)

    if lines is not None:
        src_clone = src.copy()

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(img=src_clone, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
            pass
        pass

        plt_show(src_clone, title='Lines with HoughP, threshold: ' + str(threshold))
    pass

    # Detect circles
    # blur = cv.GaussianBlur(src=image_gray, ksize=(5, 5), sigmaX=1.5)
    threshold = 200
    min_votes = 100

    circles = cv.HoughCircles(image=image_gray, method=cv.HOUGH_GRADIENT, dp=2, minDist=20,
                              param1=threshold, param2=min_votes, minRadius=15, maxRadius=50)

    if circles is not None:
        src_clone = src.copy()
        for circle in circles:
            for x1, y1, r in circle:
                cv.circle(img=src_clone, center=(x1, y1), radius=int(r), color=(0, 255, 0), thickness=2)
            pass

        plt_show(src_clone, title='Circles with HoughP, threshold: ' + str(threshold) + ', min_votes=' + str(min_votes))

    pass

    # Get the contours
    src_clone = src.copy()
    contours, _ = cv.findContours(image=image_gray, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)
    image_contours = cv.drawContours(image=src_clone, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2)

    plt_show(image_contours, title='Contours with RETR_LIST')



