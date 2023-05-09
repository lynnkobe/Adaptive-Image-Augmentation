from Deeplab.Deeplab_Class import DeepLab
import torch
import cv2 as cv
import numpy as np
from Deeplab.SegmentIndex import SegmentationMetric
import os
from Deeplab.DeeplabDataset import Dataset
import torch.utils.data as Data


# 测试集用网络预测并输出iou等五个评价指标的平均值、也可以输入训练集，看训练集的训练效果
# 输入：weight路径及名字；输出：分割质量指标IOU等
def Deeplab_Test(weight_file_name, testset_path, save_path = "Deeplab/AppleRustSet/Test_Predict_label/"):
    CLASS_NUM = 2
    WEIGHTS = weight_file_name  # 训练好的权重路径
    deeplab_test = DeepLab(num_classes=CLASS_NUM,
                 backbone='mobilenet',
                 output_stride=16,
                 sync_bn=False,
                 freeze_bn=False)
    deeplab_test.load_state_dict(torch.load(WEIGHTS))
    deeplab_test.eval()
    # cpu-cuda
    deeplab_test.cuda()

    Pa = []
    CPA = []
    MPA = []
    IOU = []
    MIOU = []

    test_data = Dataset(testset_path[0], testset_path[1])
    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False)

    for index, data in enumerate(test_loader, 0):
        image, label = data

        # cpu-cuda
        image = image.cuda()

        output = deeplab_test(image)

        # 预测结果
        output = torch.squeeze(output)
        output = output.argmax(dim=0)
        # cuda-cpu
        output = output.cpu().numpy()

        # 标签结果
        label = torch.squeeze(label)
        # cuda-cpu
        label = label.cpu().numpy()

        # 维度还原
        predict = cv.resize(np.uint8(output), (512, 512))
        label = cv.resize(np.uint8(label), (512, 512))



        # 保存预测结果标签图，可取消注释开启功能，默认将标签保存到Test_Predict_label下
        # predict_filename = os.listdir('Deeplab/AppleRustSet/Test(origin)')[index]
        # cv.imwrite(save_path + predict_filename, predict * 255)
        # np.savetxt('shiyan1.txt', predict, fmt='%d')

        # np.savetxt('target2.txt', target, fmt='%d')

        # =======================用SegmentIndex方法来计算miou及pa等指标=======================================
        metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几
        hist = metric.addBatch(np.array(predict, dtype=np.uint8), np.array(label, dtype=np.uint8))
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()[1]
        mpa = metric.meanPixelAccuracy()
        iou = metric.IntersectionOverUnion()[1]
        miou = metric.meanIntersectionOverUnion()

        Pa.append(pa)
        CPA.append(cpa)
        MPA.append(mpa)
        IOU.append(iou)
        MIOU.append(miou)
        # print("pa_{0}:{1}".format(index, format(Pa[index], ".6f")))
        # print("cpa_{0}:{1}".format(index, format(CPA[index], ".6f")))
        # print("mpa_{0}:{1}".format(index, format(MPA[index], ".6f")))
        # print("iou_{0}:{1}".format(index, format(IOU[index], ".6f")))
        # print("miou_{0}:{1}".format(index, format(MIOU[index], ".6f")))

    # 将各个集合中nan转为0
    Pa = np.nan_to_num(Pa)
    CPA = np.nan_to_num(CPA)
    MPA = np.nan_to_num(MPA)
    IOU = np.nan_to_num(IOU)
    MIOU = np.nan_to_num(MIOU)

    print("average_pa:{}".format(format(np.mean(Pa), ".6f")))
    print("average_cpa:{}".format(format(np.mean(CPA), ".6f")))
    print("average_mpa:{}".format(format(np.mean(MPA), ".6f")))
    print("average_iou:{}".format(format(np.mean(IOU), ".10f")))
    print("average_miou:{}".format(format(np.mean(MIOU), ".6f")))

    return np.mean(Pa), np.mean(CPA), np.mean(MPA), np.mean(IOU), np.mean(MIOU)


if __name__=='__main__':
# if 0:
    weight_file_name="./Deeplab/deeplab_weights/test_weights/deeplab_weights(batch32_lr0.001_Adam(rust)).pth"
    testset_path = ['Deeplab/AppleRustSet/Test(origin)', 'Deeplab/AppleRustSet/Test_label(origin)']
    # Deeplab_Test(weight_file_name,"./train.txt")
    r0 = Deeplab_Test(weight_file_name, testset_path, save_path="Deeplab/AppleRustSet/leaf_Predict_label/") [3]
    print(r0)
