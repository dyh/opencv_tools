import cv2 as cv

from image_filtering import show_filtering
from image_outline import show_outline
from image_transformation import show_transformation
from utils import plt_show
from image_color import show_hsv


if __name__ == '__main__':
    # file_path = './images/000000050145.jpg'
    file_path = './images/000000507081.jpg'
    # file_path = './images/000000024021.jpg'
    # file_path = './images/1130780078.bmp'

    origin = cv.imread(file_path)
    origin = origin[:, :, [2, 1, 0]]

    # x, y = origin.shape[0:2]
    # origin = cv2.resize(origin, (int(y / 3), int(x / 3)))
    plt_show(image=origin, title='Origin')

    # --------------------图像色彩--------------------

    # 转换成HSV色彩空间
    show_hsv(origin)

    # --------------------图像变换--------------------
    show_transformation(origin)

    # --------------------图像过滤--------------------
    show_filtering(origin)

    # --------------------提取直线、轮廓、区域--------------------
    show_outline(origin)

    print('done')


