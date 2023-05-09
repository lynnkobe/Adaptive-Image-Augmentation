
import numpy as np
import cv2
import pandas as pd
import os
from Deeplab.Deeplab_Class import DeepLab
import torch


def compute_character(path, deeplab):
    path = path.replace('\\', '/')
    img = cv2.imread(path, flags=1)
    input = cv2.resize(img, (224, 224))
    input = input / 255.0  # 归一化输入
    input = torch.tensor(input).permute(2, 0, 1)
    input = input.unsqueeze(0).cuda().float()

    label = deeplab(input)

    # 预测结果
    output = torch.squeeze(label)
    output = output.argmax(dim=0)
    output = output.cpu().numpy()
    output = cv2.resize(np.uint8(output), (512, 512))
    output = output * 255

    # 查找所有白色块的轮廓
    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化总面积和加权重心的坐标
    total_area = 0
    weighted_cx = 0
    weighted_cy = 0

    # 遍历所有白色块的轮廓
    for contour in contours:
        area = cv2.contourArea(contour)  # 计算轮廓的面积

        # 忽略面积小于100的轮廓，这些可能是噪点或者文本
        if area < 100:
            continue

        # 计算轮廓的重心
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 更新总面积和加权重心的坐标
        total_area += area
        weighted_cx += area * cx
        weighted_cy += area * cy

        # 在图像中标记重心
        # cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)

    # 计算所有白色块重心的平均值
    if total_area > 0:
        avg_cx = int(weighted_cx / total_area)
        avg_cy = int(weighted_cy / total_area)

    else:  # 没有rust斑点的情况下，重心位于中心
        avg_cx = 0.5
        avg_cy = 0.5
        # 在图像中标记综合的重心
        # cv2.circle(output, (avg_cx, avg_cy), 6, (0, 255, 0), -1)

    # 打印病斑总面积和加权平均重心坐标的相对比例
    height, width = output.shape[:2]
    image_area = height * width

    area = total_area / image_area
    center_x = avg_cx / width
    center_y = avg_cy / height

    # cv2.imwrite('Deeplab/AppleRustSet/rust_label_point/2.png', output)

    # 创建一个掩膜，将黑色像素点标记为0，其他像素点标记为1
    mask = cv2.threshold(output, 0, 1, cv2.THRESH_BINARY)[1]
    number_notblack_pixel = np.mean(mask)

    if number_notblack_pixel>0:
        # 计算R通道的平均值
        avg_r = np.mean(img[:, :, 2] * mask) / number_notblack_pixel /255
        avg_g = np.mean(img[:, :, 1] * mask) / number_notblack_pixel /255
        avg_b = np.mean(img[:, :, 0] * mask) / number_notblack_pixel /255
    else:
        avg_r = 0
        avg_g = 0
        avg_b = 0
    # # 定义像素点计数器和RGB值总和变量
    # count = 0
    # sum_r, sum_g, sum_b = 0, 0, 0
    # height, width = output.shape[:2]
    # # 遍历分割图像的每个像素
    # for x in range(height):
    #     for y in range(width):
    #         # 如果像素不是黑色，统计RGB值
    #         if np.all(output[x, y] != 0):
    #             b, g, r = img[x, y]
    #             count += 1
    #             sum_r += r
    #             sum_g += g
    #             sum_b += b
    # # 计算像素点的RGB值的均值,并除以255求占比
    # if count > 0:
    #     avg_r = sum_r // count / 255
    #     avg_g = sum_g // count / 255
    #     avg_b = sum_b // count / 255
    #     # print(f"{filename}: ({avg_r}, {avg_g}, {avg_b})")
    # else:
    #     print(f"{path}: no colored pixels")

    # print('{0} {1} {2} {3} {4} {5}'.format(area,center_x,center_y,avg_r,avg_g,avg_b))
    return np.array([area, center_x, center_y, avg_r, avg_g, avg_b])


if __name__=='__main__':
    path = 'Deeplab/AppleRustSet/State/1.png'
    compute_character(path)

