import cv2
import numpy as np

from utils import plt_save


def show_transformation(src):
    # 用形态学滤波器腐蚀和膨胀图像
    element_3x3 = np.ones((3, 3), np.uint8)

    # 腐蚀 3x3
    eroded = cv2.erode(src=src, kernel=element_3x3)
    plt_save(image=eroded, title='eroded')

    # 膨胀 3x3 3次
    dilated = cv2.dilate(src=src, kernel=element_3x3, iterations=3)
    plt_save(image=dilated, title='dilated 3 times')

    # 腐蚀 7x7
    element_7x7 = np.ones((7, 7), np.uint8)
    eroded_7x7 = cv2.erode(src=src, kernel=element_7x7, iterations=1)
    plt_save(image=eroded_7x7, title='eroded 7x7')

    # 腐蚀 3x3 3次
    eroded_3 = cv2.erode(src=src, kernel=element_3x3, iterations=3)
    plt_save(image=eroded_3, title='eroded 3 times')

    # 用形态学滤波器开启和闭合图像
    image_gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    # plt_save(image=image_gray, title='image_gray')

    # Close the image
    element_5x5 = np.ones((5, 5), np.uint8)

    closed = cv2.morphologyEx(src=image_gray, op=cv2.MORPH_CLOSE, kernel=element_5x5)
    plt_save(image=closed, title='closed')

    # Open the image
    opened = cv2.morphologyEx(src=image_gray, op=cv2.MORPH_OPEN, kernel=element_5x5)
    plt_save(image=opened, title='opened')

    closed = cv2.morphologyEx(src=image_gray, op=cv2.MORPH_CLOSE, kernel=element_5x5)
    closed_opened = cv2.morphologyEx(src=closed, op=cv2.MORPH_OPEN, kernel=element_5x5)
    plt_save(image=closed_opened, title='Closed -> Opened')

    opened = cv2.morphologyEx(src=image_gray, op=cv2.MORPH_OPEN, kernel=element_5x5)
    opened_closed = cv2.morphologyEx(src=opened, op=cv2.MORPH_CLOSE, kernel=element_5x5)
    plt_save(image=opened_closed, title='Opened -> Closed')

    # 在灰度图像中应用形态学运算
    edge = cv2.morphologyEx(src=image_gray, op=cv2.MORPH_GRADIENT, kernel=element_3x3)
    plt_save(image=255 - edge, title='Gradient | Edge')

    # Apply threshold to obtain a binary image
    threshold = 80
    _, thresh_binary = cv2.threshold(src=edge, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
    plt_save(image=thresh_binary, title='Gradient | Edge -> Thresh Binary | Edge')

    # 7x7 Black Top-hat Image
    black_hat = cv2.morphologyEx(src=image_gray, op=cv2.MORPH_BLACKHAT, kernel=element_7x7)
    plt_save(image=255 - black_hat, title='7x7 Black Top-hat')

    # Apply threshold to obtain a binary image
    threshold = 25
    _, thresh_binary = cv2.threshold(src=black_hat, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
    plt_save(image=255 - thresh_binary, title='7x7 Black Top-hat -> Thresh Binary | Edge')

    # Apply the black top-hat transform using a 7x7 structuring element
    closed = cv2.morphologyEx(src=thresh_binary, op=cv2.MORPH_CLOSE, kernel=element_7x7)
    plt_save(image=255 - closed, title='7x7 Black Top-hat -> Closed')

    pass
