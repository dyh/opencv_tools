import cv2
import numpy as np

from utils import plt_save


def show_outline(src):
    contours = cv2.Canny(image=src, threshold1=125, threshold2=350)
    plt_save(255 - contours, title='Canny Contours')

    image_gray = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2GRAY)

    contours = cv2.Canny(image=image_gray, threshold1=125, threshold2=350)
    plt_save(255 - contours, title='Canny Contours Gray')

    # Hough tranform for line detection
    theta = np.pi / 180
    threshold = 50
    lines = cv2.HoughLinesP(image=contours, rho=1, theta=theta, threshold=threshold)

    if lines is not None:
        src_clone = src.copy()

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img=src_clone, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
            pass
        pass

        plt_save(src_clone, title='Lines with HoughP, threshold: ' + str(threshold))
    pass

    # Detect circles
    # blur =  cv2.GaussianBlur(src=image_gray, ksize=(5, 5), sigmaX=1.5)
    threshold = 200
    min_votes = 100

    circles = cv2.HoughCircles(image=image_gray, method=cv2.HOUGH_GRADIENT, dp=2, minDist=20,
                               param1=threshold, param2=min_votes, minRadius=15, maxRadius=50)

    if circles is not None:
        src_clone = src.copy()
        for circle in circles:
            for x1, y1, r in circle:
                cv2.circle(img=src_clone, center=(x1, y1), radius=int(r), color=(0, 255, 0), thickness=2)
            pass

        plt_save(src_clone, title='Circles with HoughP, threshold: ' + str(threshold) + ', min_votes=' + str(min_votes))

    pass

    # Get the contours
    src_clone = src.copy()
    contours, _ = cv2.findContours(image=image_gray, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    image_contours = cv2.drawContours(image=src_clone, contours=contours, contourIdx=-1, color=(255, 255, 255),
                                      thickness=2)

    plt_save(image_contours, title='Contours with RETR_LIST')
