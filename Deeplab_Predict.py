from Deeplab.Deeplab_Class import DeepLab
import torch
import cv2 as cv
import numpy as np
import os
from Deeplab.DeeplabDataset import Dataset
import torch.utils.data as Data


# 测试集用网络预测并输出iou等五个评价指标的平均值、也可以输入训练集，看训练集的训练效果
# 输入：weight路径及名字；输出：分割质量指标IOU等
def Deeplab_Predict(weight_file_name, testset_path, save_path = "Deeplab/AppleRustSet/Test_Predict_label/"):
    CLASS_NUM = 2
    WEIGHTS = weight_file_name  # 训练好的权重路径
    deeplab_predict = DeepLab(num_classes=CLASS_NUM,
                 backbone='mobilenet',
                 output_stride=16,
                 sync_bn=False,
                 freeze_bn=False)
    deeplab_predict.load_state_dict(torch.load(WEIGHTS))
    deeplab_predict.eval()
    # cpu-cuda
    deeplab_predict.cuda()

    test_data = Dataset(testset_path[0], testset_path[1])
    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False)

    for index, data in enumerate(test_loader, 0):
        image, label = data

        # cpu-cuda
        image = image.cuda()

        output = deeplab_predict(image)

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
        predict_filename = os.listdir('Deeplab/AppleRustSet/Test(origin)')[index]
        cv.imwrite(save_path + predict_filename, predict * 255)
        # np.savetxt('shiyan1.txt', predict, fmt='%d')


if __name__=='__main__':
# if 0:
    weight_file_name = "./Deeplab/deeplab_weights/test_weights/deeplab_weights(batch32_lr0.001_Adam).pth"
    testset_path = ['Deeplab/AppleRustSet/Test(origin)', 'Deeplab/AppleRustSet/Test_label(origin)']
    # Deeplab_Test(weight_file_name,"./train.txt")
    Deeplab_Predict(weight_file_name, testset_path, save_path="Deeplab/AppleRustSet/leaf_Predict_label/")

