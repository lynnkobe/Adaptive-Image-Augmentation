import cv2 as cv
import numpy as np

# 选择图像更新方式：
def do_img_aug(input_img_path, aug_num):
    input_img = cv.imread(input_img_path, flags=1)  # 1-三通道/0-灰度图
    # cv.imshow('Original', input_img)
    # cv.waitKey(0)
    if aug_num == 0:
        # 1、原图
        return input_img

    elif aug_num == 1:
        # 2、垂直翻转
        return cv.flip(input_img, 0)

    if aug_num == 2:
        # 3、水平翻转
        return cv.flip(input_img, 1)

    elif aug_num == 3:
        # 4、水平和垂直翻转
        return cv.flip(input_img, -1)

    elif aug_num == 4:
        # 5、旋转
        height, width = input_img.shape[:2]  # 图片的高度和宽度
        theta = 30  # 顺时针旋转角度，单位为角度
        x0, y0 = width // 2, height // 2  # 以图像中心作为旋转中心
        MAR = cv.getRotationMatrix2D((x0, y0), theta, 1.0)
        dst = cv.warpAffine(input_img, MAR, (width, height), borderValue=(255, 255, 255))  # 设置白色填充
        return dst

    # elif aug_num == 5:
    #     # 6、缩放(但像素变了）
    #     height, width = input_img.shape[:2]  # 图片的高度和宽度
    #     imgZoom = cv.resize(input_img, None, fx=1.75, fy=1.25, interpolation=cv.INTER_AREA)
    #     imgZoom = cv.resize(imgZoom, (512, 512))
    #     return imgZoom

    elif aug_num == 5:
        # 7、错切变换
        height, width = input_img.shape[:2]  # 图片的高度和宽度
        MAS = np.float32([[1, 0.2, 0], [0, 1, 0]])  # 构造错切变换矩阵
        imgShear = cv.warpAffine(input_img, MAS, (width, height))
        return imgShear

    elif aug_num == 6:
        # 8、裁剪
        xmin, ymin, w, h = 25, 25, 487, 487  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
        imgCrop = input_img[ymin:ymin + h, xmin:xmin + w].copy()  # 切片获得裁剪后保留的图像区域
        imgCrop = cv.resize(imgCrop, (512, 512))
        return imgCrop

    elif aug_num == 7:
        # 9、噪声
        mu, sigma = 0.0, 20.0
        noiseGause = np.random.normal(mu, sigma, input_img.shape)
        imgGaussNoise = input_img + noiseGause
        imgGaussNoise = np.uint8(cv.normalize(imgGaussNoise, None, 0, 255, cv.NORM_MINMAX))  # 归一化为 [0,255]
        return imgGaussNoise

    # elif aug_num == 9:
    #     # 10、色彩平衡
    #     maxG = 128  # 修改颜色通道最大值，0<=maxG<=255
    #     lutHalf = np.array([int(i * maxG / 255) for i in range(256)]).astype("uint8")
    #     lutEqual = np.array([i for i in range(256)]).astype("uint8")
    #     lut3HalfB = np.dstack((lutHalf, lutEqual, lutEqual))  # (1,256,3), B_half/BGR
    #     blendHalfB = cv.LUT(input_img, lut3HalfB)  # B 通道衰减 50%
    #     return blendHalfB



# 选择图像标签更新方式：
def do_img_aug_label(input_img_path, aug_num):
    input_img = cv.imread(input_img_path, flags=0)  # 1-三通道/0-灰度图
    # cv.imshow('Original_label', input_img)
    # cv.waitKey(0)
    if aug_num == 0:
        # 1、原图
        return input_img

    elif aug_num == 1:
        # 2、垂直翻转
        return cv.flip(input_img, 0)

    if aug_num == 2:
        # 3、水平翻转
        return cv.flip(input_img, 1)

    elif aug_num == 3:
        # 4、水平和垂直翻转
        return cv.flip(input_img, -1)

    elif aug_num == 4:
        # 5、旋转
        height, width = input_img.shape[:2]  # 图片的高度和宽度
        theta = 30  # 顺时针旋转角度，单位为角度
        x0, y0 = width // 2, height // 2  # 以图像中心作为旋转中心
        MAR = cv.getRotationMatrix2D((x0, y0), theta, 1.0)
        dst = cv.warpAffine(input_img, MAR, (width, height), borderValue=(0, 0, 0))  # 设置黑色填充
        return dst

    # elif aug_num == 5:
    #     # 6、缩放(但像素变了）
    #     height, width = input_img.shape[:2]  # 图片的高度和宽度
    #     imgZoom = cv.resize(input_img, None, fx=1.75, fy=1.25, interpolation=cv.INTER_AREA)
    #     imgZoom = cv.resize(imgZoom, (512, 512))
    #     return imgZoom

    elif aug_num == 5:
        # 7、错切变换
        height, width = input_img.shape[:2]  # 图片的高度和宽度
        MAS = np.float32([[1, 0.2, 0], [0, 1, 0]])  # 构造错切变换矩阵
        imgShear = cv.warpAffine(input_img, MAS, (width, height))
        return imgShear

    elif aug_num == 6:
        # 8、裁剪
        xmin, ymin, w, h = 25, 25, 487, 487  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
        imgCrop = input_img[ymin:ymin + h, xmin:xmin + w].copy()  # 切片获得裁剪后保留的图像区域
        imgCrop = cv.resize(imgCrop, (512, 512))
        return imgCrop

    elif aug_num == 7:
        # 9、噪声
        # mu, sigma = 0.0, 20.0
        # noiseGause = np.random.normal(mu, sigma, input_img.shape)
        # imgGaussNoise = input_img + noiseGause
        # imgGaussNoise = np.uint8(cv.normalize(imgGaussNoise, None, 0, 255, cv.NORM_MINMAX))  # 归一化为 [0,255]
        return input_img

    # elif aug_num == 9:
    #     # # 10、色彩平衡
    #     # maxG = 128  # 修改颜色通道最大值，0<=maxG<=255
    #     # lutHalf = np.array([int(i * maxG / 255) for i in range(256)]).astype("uint8")
    #     # lutEqual = np.array([i for i in range(256)]).astype("uint8")
    #     # lut3HalfB = np.dstack((lutHalf, lutEqual, lutEqual))  # (1,256,3), B_half/BGR
    #     # blendHalfB = cv.LUT(input_img, lut3HalfB)  # B 通道衰减 50%
    #     return input_img