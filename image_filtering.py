import cv2
import numpy as np

from utils import plt_save


def show_filtering(src):
    # 低通滤波
    # Blur the image with a mean filter
    blur = cv2.blur(src=src, ksize=(5, 5))
    plt_save(image=blur, title='Mean filtered (5x5)')

    # Blur the image with a mean filter 9x9
    blur = cv2.blur(src=src, ksize=(9, 9))
    plt_save(image=blur, title='Mean filtered (9x9)')

    blur = cv2.GaussianBlur(src=src, ksize=(9, 9), sigmaX=1.5)
    plt_save(image=blur, title='Gaussian filtered Image (9x9)')

    gauss = cv2.getGaussianKernel(ksize=9, sigma=1.5, ktype=cv2.CV_32F)
    print('GaussianKernel 1.5 = [', end='')
    for item in gauss:
        print(item, end='')
    pass
    print(']')

    gauss = cv2.getGaussianKernel(ksize=9, sigma=-1, ktype=cv2.CV_32F)
    print('GaussianKernel -1 = [', end='')
    for item in gauss:
        print(item, end='')
    pass
    print(']')

    # 缩减 采样
    blur = cv2.GaussianBlur(src=src, ksize=(11, 11), sigmaX=1.75)
    resized1 = cv2.resize(src=blur, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    plt_save(image=resized1, title='resize CUBIC 0.25')

    # resizing with NN
    resized2 = cv2.resize(src=resized1, dsize=(0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    plt_save(image=resized2, title='resize NEAREST x4')

    # resizing with bilinear
    resized3 = cv2.resize(src=resized1, dsize=(0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    plt_save(image=resized3, title='resize LINEAR x4')

    # 中值滤波
    median_blur = cv2.medianBlur(src=src, ksize=5)
    plt_save(image=median_blur, title='Median filtered')

    # 定向滤波器
    image_gray = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2GRAY)

    # Compute Sobel X derivative
    sobel_x = cv2.Sobel(src=image_gray, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3, scale=0.4, delta=128,
                        borderType=cv2.BORDER_DEFAULT)
    plt_save(image=sobel_x, title='Sobel X')

    # Compute Sobel Y derivative
    sobel_y = cv2.Sobel(src=image_gray, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3, scale=0.4, delta=128,
                        borderType=cv2.BORDER_DEFAULT)
    plt_save(image=sobel_y, title='Sobel Y')

    # Compute norm of Sobel
    sobel_x = cv2.Sobel(src=image_gray, ddepth=cv2.CV_16S, dx=1, dy=0)
    sobel_y = cv2.Sobel(src=image_gray, ddepth=cv2.CV_16S, dx=0, dy=1)

    sobel_1 = abs(sobel_x) + abs(sobel_y)
    plt_save(image=sobel_1, title='abs Sobel X+Y')

    sobel_2 = cv2.convertScaleAbs(sobel_x) + cv2.convertScaleAbs(sobel_y)
    plt_save(image=sobel_2, title='cv2.convertScaleAbs Sobel X+Y')

    # Compute Sobel X derivative (7x7)
    sobel_x_7x7 = cv2.Sobel(src=image_gray, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=7, scale=0.001, delta=128)
    plt_save(image=sobel_x_7x7, title='Sobel X (7x7)')

    uint8_sobel_1 = np.uint8(sobel_1)
    plt_save(image=uint8_sobel_1, title='uint8 sobel_1')

    uint8_sobel_2 = np.uint8(sobel_2)
    plt_save(image=uint8_sobel_2, title='uint8 sobel_2')

    int8_sobel_1 = np.int8(sobel_1)
    plt_save(image=int8_sobel_1, title='int8 sobel_1')

    int8_sobel_2 = np.int8(sobel_2)
    plt_save(image=int8_sobel_2, title='int8 sobel_2')

    # Apply threshold to Sobel norm (low threshold value)
    _, thresh_binary = cv2.threshold(src=uint8_sobel_1, thresh=255, maxval=255, type=cv2.THRESH_BINARY)
    plt_save(image=thresh_binary, title='Binary Sobel (low)  cv2.threshold uint8_sobel_1')

    _, thresh_binary = cv2.threshold(src=uint8_sobel_2, thresh=255, maxval=255, type=cv2.THRESH_BINARY)
    plt_save(image=thresh_binary, title='Binary Sobel (low)  cv2.threshold uint8_sobel_2')

    # Apply threshold to Sobel norm (high threshold value)
    _, thresh_binary = cv2.threshold(src=uint8_sobel_1, thresh=190, maxval=255, type=cv2.THRESH_BINARY)
    plt_save(image=thresh_binary, title='Binary Sobel Image (high)  cv2.threshold uint8_sobel_1')

    _, thresh_binary = cv2.threshold(src=uint8_sobel_2, thresh=190, maxval=255, type=cv2.THRESH_BINARY)
    plt_save(image=thresh_binary, title='Binary Sobel Image (high)  cv2.threshold uint8_sobel_2')

    add_weighted_sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    plt_save(image=add_weighted_sobel, title='cv2.addWeighted abs')

    # down-sample and up-sample the image
    reduced = cv2.pyrDown(src=src)
    rescaled = cv2.pyrUp(src=reduced)
    plt_save(image=rescaled, title='Rescaled')

    # down-sample and up-sample the image
    reduced = cv2.pyrDown(src=image_gray)
    rescaled = cv2.pyrUp(src=reduced)

    w, h = src.shape[0:2]
    rescaled = cv2.resize(rescaled, (h, w))

    plt_save(image=rescaled, title='Rescaled')

    subtract = cv2.subtract(src1=rescaled, src2=image_gray)
    subtract = np.uint8(subtract)
    plt_save(image=subtract, title='cv2.subtract')

    gauss05 = cv2.GaussianBlur(src=image_gray, ksize=(0, 0), sigmaX=0.5)
    gauss15 = cv2.GaussianBlur(src=image_gray, ksize=(0, 0), sigmaX=1.5)
    subtract = cv2.subtract(src1=gauss15, src2=gauss05, dtype=cv2.CV_16S)
    subtract = np.uint8(subtract)
    plt_save(image=subtract, title='cv2.subtract gauss15 - gauss05')

    gauss20 = cv2.GaussianBlur(src=image_gray, ksize=(0, 0), sigmaX=2.0)
    gauss22 = cv2.GaussianBlur(src=image_gray, ksize=(0, 0), sigmaX=2.2)
    subtract = cv2.subtract(src1=gauss22, src2=gauss20, dtype=cv2.CV_32F)
    subtract = np.uint8(subtract)
    plt_save(image=subtract, title='cv2.subtract gauss22 - gauss20')
