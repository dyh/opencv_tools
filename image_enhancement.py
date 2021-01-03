import random
import sys

import cv2
import numpy as np

from utils import plt_save


def show_enhancement(src):
    balance_img1 = white_balance(src)
    plt_save(balance_img1, 'White Balance 1')

    # 灰度世界算法
    balance_img2 = grey_world(src)
    plt_save(balance_img2, 'White Balance 2')

    # 直方图均衡化
    balance_img3 = his_equl_color(src)
    plt_save(balance_img3, 'White Balance 3')

    # 视网膜-大脑皮层(Retinex)增强算法
    # 单尺度Retinex
    ssr1 = single_scale_retinex(src, 300)
    ssr1[ssr1 < 0] = 0
    plt_save(ssr1, 'Single Scale Retinex 1')

    ssr2 = s_s_r(src)
    plt_save(ssr2, 'Single Scale Retinex 2')

    # 多尺度Retinex
    msr1 = multi_scale_retinex(src, [15, 80, 250])
    msr1[msr1 < 0] = 0
    plt_save(msr1, 'Multi Scale Retinex 1')

    msr2 = m_s_r(src, sigma_list=[15, 80, 250])
    plt_save(msr2, 'Multi Scale Retinex 2')

    msrcr1 = m_s_r_c_r(src, sigma_list=[15, 80, 250])
    plt_save(msrcr1, 'Multi Scale Retinex With Color Restoration 1')

    # 自动白平衡 AWB
    awb = automatic_white_balance(src)
    plt_save(awb, 'Automatic White Balance')

    # 自动色彩均衡 ACE
    balance_img4 = automatic_color_equalization(src)
    plt_save(balance_img4, 'Automatic Color Equalization')


def white_balance(src):
    r, g, b = cv2.split(src)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]

    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg

    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    return cv2.merge([b, g, r])


# 灰度世界算法
def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    avg_b = np.average(nimg[0])
    avg_g = np.average(nimg[1])
    avg_r = np.average(nimg[2])

    avg = (avg_b + avg_g + avg_r) / 3

    nimg[0] = np.minimum(nimg[0] * (avg / avg_b), 255)
    nimg[1] = np.minimum(nimg[1] * (avg / avg_g), 255)
    nimg[2] = np.minimum(nimg[2] * (avg / avg_r), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


# 直方图均衡化
def his_equl_color(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])  # equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq


# 视网膜-大脑皮层(Retinex)增强算法
def single_scale_retinex(img, sigma):
    temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(temp == 0, 0.01, temp)
    retinex = np.log10(img + 0.01) - np.log10(gaussian)
    return retinex


def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        retinex = single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex


def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration_temp = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration_temp


def multi_scale_retinex_with_color_restoration(img, sigma_list, g, b, alpha, beta):
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_msrcr = g * (img_retinex * img_color + b)
    return img_msrcr


def simplest_color_balance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]

    low_val = 0.0
    high_val = 0.0

    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0

        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img


def touint8(img):
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
                       (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
    img = np.uint8(np.minimum(np.maximum(img, 0), 255))
    return img


def s_s_r(img, sigma=300):
    ssr = single_scale_retinex(img, sigma)
    ssr = touint8(ssr)
    return ssr


def m_s_r(img, sigma_list):
    if sigma_list is None:
        sigma_list = [15, 80, 250]
    msr = multi_scale_retinex(img, sigma_list)
    msr = touint8(msr)
    return msr


def m_s_r_c_r(img, sigma_list, g=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    if sigma_list is None:
        sigma_list = [15, 80, 250]
    msrcr = multi_scale_retinex_with_color_restoration(img, sigma_list, g, b, alpha, beta)
    msrcr = touint8(msrcr)
    msrcr = simplest_color_balance(msrcr, low_clip, high_clip)
    return msrcr


# 自动白平衡(AWB)
def automatic_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


# 自动色彩均衡(ACE)
# 饱和函数
def calc_saturation(diff, slope, limit):
    ret = diff * slope
    if ret > limit:
        ret = limit
    elif ret < (-limit):
        ret = -limit
    return ret


def automatic_color_equalization(nimg, slope=10, limit=1000, samples=500):
    nimg = nimg.transpose(2, 0, 1)

    # Convert input to an ndarray with column-major memory order(仅仅是地址连续，内容和结构不变)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)

    width = nimg.shape[2]
    height = nimg.shape[1]

    cary = []

    # 随机产生索引
    for i in range(0, samples):
        _x = random.randint(0, width) % width
        _y = random.randint(0, height) % height

        dict_temp = {"x": _x, "y": _y}
        cary.append(dict_temp)
        pass

    mat = np.zeros((3, height, width), float)

    r_max = sys.float_info.min
    r_min = sys.float_info.max

    g_max = sys.float_info.min
    g_min = sys.float_info.max

    b_max = sys.float_info.min
    b_min = sys.float_info.max

    for i in range(height):
        for j in range(width):
            r = nimg[0, i, j]
            g = nimg[1, i, j]
            b = nimg[2, i, j]

            r_rscore_sum = 0.0
            g_rscore_sum = 0.0
            b_rscore_sum = 0.0
            denominator = 0.0

            for _dict in cary:
                _x = _dict["x"]  # width
                _y = _dict["y"]  # height

                # 计算欧氏距离
                dist = np.sqrt(np.square(_x - j) + np.square(_y - i))
                if dist < height / 5:
                    continue

                _sr = nimg[0, _y, _x]
                _sg = nimg[1, _y, _x]
                _sb = nimg[2, _y, _x]

                r_rscore_sum += calc_saturation(int(r) - int(_sr), slope, limit) / dist
                g_rscore_sum += calc_saturation(int(g) - int(_sg), slope, limit) / dist
                b_rscore_sum += calc_saturation(int(b) - int(_sb), slope, limit) / dist

                denominator += limit / dist

            r_rscore_sum = r_rscore_sum / denominator
            g_rscore_sum = g_rscore_sum / denominator
            b_rscore_sum = b_rscore_sum / denominator

            mat[0, i, j] = r_rscore_sum
            mat[1, i, j] = g_rscore_sum
            mat[2, i, j] = b_rscore_sum

            if r_max < r_rscore_sum:
                r_max = r_rscore_sum
            if r_min > r_rscore_sum:
                r_min = r_rscore_sum

            if g_max < g_rscore_sum:
                g_max = g_rscore_sum
            if g_min > g_rscore_sum:
                g_min = g_rscore_sum

            if b_max < b_rscore_sum:
                b_max = b_rscore_sum
            if b_min > b_rscore_sum:
                b_min = b_rscore_sum

    for i in range(height):
        for j in range(width):
            nimg[0, i, j] = (mat[0, i, j] - r_min) * 255 / (r_max - r_min)
            nimg[1, i, j] = (mat[1, i, j] - g_min) * 255 / (g_max - g_min)
            nimg[2, i, j] = (mat[2, i, j] - b_min) * 255 / (b_max - b_min)

    return nimg.transpose([1, 2, 0]).astype(np.uint8)
