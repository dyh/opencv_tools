import cv2 as cv
import numpy as np

from utils import plt_show


def show_hsv(src):
    # 转换成HSV色彩空间
    hsv = cv.cvtColor(src=src, code=cv.COLOR_BGR2HSV)

    # 转回BGR
    # bgr = cv.cvtColor(src=hsv, code=cv.COLOR_HSV2BGR)
    # plt_show(bgr, 'BGR')

    h, s, v = cv.split(hsv)

    # 色度/色调
    plt_show(image=h, title='Hue')
    # 饱和度
    plt_show(image=s, title='Saturation')
    # 纯度/亮度
    plt_show(image=v, title='Value')

    # 固定色度h
    h_new = np.full_like(a=h, fill_value=255)
    merge = cv.merge([h_new, s, v])
    plt_show(merge, 'Fixed Hue')

    # 固定饱和度s
    s_new = np.full_like(a=s, fill_value=255)
    merge = cv.merge([h, s_new, v])
    plt_show(merge, 'Fixed Saturation')

    # 固定亮度v
    v_new = np.full_like(a=v, fill_value=255)
    merge = cv.merge([h, s, v_new])
    plt_show(merge, 'Fixed Value')

    # 固定色度h + 固定饱和度s
    merge = cv.merge([h_new, s_new, v])
    plt_show(merge, 'Fixed Hue & Saturation')

    # 固定色度h + 固定亮度v
    merge = cv.merge([h_new, s, v_new])
    plt_show(merge, 'Fixed Hue & Value')

    # 固定饱和度s + 固定亮度v
    merge = cv.merge([h, s_new, v_new])
    plt_show(merge, 'Fixed Saturation & Value')
