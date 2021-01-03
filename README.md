# 图像预处理 pipeline 工具

一键预览 OpenCV 60 种图像效果

### 视频

bilibili

[![bilibili](https://github.com/dyh/opencv_tools/blob/main/cover.jpg?raw=true)](https://www.bilibili.com/video/BV1Cy4y127P7/ "bilibili")


### 图像色彩

image_color.py

- 色度/色调
- 饱和度
- 纯度/亮度
- 固定饱和度s
- 固定亮度v
- 固定色度h + 固定饱和度s
- 固定色度h + 固定亮度v
- 固定饱和度s + 固定亮度v

### 图像变换

image_transformation.py

- 形态学滤波器腐蚀和膨胀图像
- 腐蚀 3x3
- 膨胀 3x3 3次
- 腐蚀 7x7
- 腐蚀 3x3 3次
- 形态学滤波器开启和闭合图像
- Close the image
- Open the image
- 灰度图像中应用形态学运算 Gradient | Edge
- Apply threshold to obtain a binary image
- 7x7 Black Top-hat Image
- Apply threshold to obtain a binary image
- Apply the black top-hat transform using a 7x7 structuring element

### 图像过滤

- Blur the image with a mean filter
- Blur the image with a mean filter 9x9
- 缩减 采样
- resizing with NN
- resizing with bilinear
- 中值滤波
- 定向滤波器
- Compute Sobel X derivative
- Compute Sobel Y derivative
- Compute norm of Sobel
- Compute Sobel X derivative (7x7)
- Apply threshold to Sobel norm (low threshold value)
- Apply threshold to Sobel norm (high threshold value)
- down-sample and up-sample the image
- down-sample and up-sample the image
- cv2.subtract
- cv2.subtract gauss15 - gauss05
- cv2.subtract gauss22 - gauss20

### 提取直线、轮廓、区域

image_outline.py

- Canny Contours
- Canny Contours Gray
- Hough tranform for line detection
- Circles with HoughP
- Get the contours, Contours with RETR_LIST

### 图像增强-白平衡等

image_enhancement.py

- 简单白平衡
- 灰度世界算法
- 直方图均衡化
- 视网膜-大脑皮层(Retinex)增强算法
- Single Scale Retinex
- Multi Scale Retinex
- Multi Scale Retinex With Color Restoration
- 自动白平衡 AWB
- 自动色彩均衡 ACE


## 运行环境

- python 3.6+，pip 20+
- pip install -r requirements.txt

    ```
    Pillow==8.0.1
    numpy==1.19.4
    opencv-python==4.4.0.46
    six==1.15.0
    matplotlib==3.3.3
    cycler==0.10.0
    kiwisolver==1.3.1
    pkg-resources==0.0.0
    pyparsing==2.4.7
    python-dateutil==2.8.1
    ```

## 如何运行

1. 克隆代码

    ```
    $ git clone https://github.com/dyh/opencv_tools.git
    ```
   
2. 进入目录

    ```
    $ cd opencv_tools
    ```

3. 创建 python 虚拟环境

    ```
    $ python3 -m venv venv
    ```

4. 激活虚拟环境

    ```
    $ source venv/bin/activate
    ```
   
5. 升级pip

    ```
    $ python -m pip install --upgrade pip
    ```

6. 安装软件包

    ```
    $ pip install -r requirements.txt
    ```

7. 在 main.py 文件中，设置要处理的图片路径 file_path，例如
 
    ```
    file_path = './images/000000050145.jpg'
    ```
   
8. 运行程序

    ```
    python main.py
    ```

9. 程序将在 output 目录下输出60张图片

