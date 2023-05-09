import os
import shutil
from Deeplab.Deeplab_Class import *
import numpy as np
from Deeplab.DeeplabDataset import *
import torch.utils.data as Data
import cv2 as cv
from Deeplab.SegmentIndex import SegmentationMetric
from time import strftime
from torch.utils.tensorboard import SummaryWriter


def ReTrain_Deeplab(epochs=0, batch_size=32):
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 定义源文件夹路径和目标文件夹路径
    origin_folder = os.path.join(current_dir, "Deeplab/AppleRustSet/Train(origin)")
    state_folder = os.path.join(current_dir, "Deeplab/AppleRustSet/State")
    origin_aug_folder = os.path.join(current_dir, "Deeplab/AppleRustSet/OriginAug")

    if os.path.exists(origin_aug_folder):
        shutil.rmtree(origin_aug_folder)
    os.mkdir(origin_aug_folder)

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(origin_aug_folder):
        os.makedirs(origin_aug_folder)

    # 将 train(origin) 文件夹中的所有文件移动到 OriginAug 文件夹中
    for filename in os.listdir(origin_folder):
        source_path = os.path.join(origin_folder, filename)
        target_path = os.path.join(origin_aug_folder, filename)
        shutil.copy(source_path, target_path)

    # 将 State 文件夹中的所有文件复制到 OriginAug 文件夹中，并在文件名前添加 "aug" 字符串
    for filename in os.listdir(state_folder):
        source_path = os.path.join(state_folder, filename)
        target_filename = "aug" + filename
        target_path = os.path.join(origin_aug_folder, target_filename)
        shutil.copy(source_path, target_path)



    # 定义源文件夹路径和目标文件夹路径
    origin_label_folder = os.path.join(current_dir, "Deeplab/AppleRustSet/Train_label(origin)")
    state_label_folder = os.path.join(current_dir, "Deeplab/AppleRustSet/State_label")
    origin_aug_label_folder = os.path.join(current_dir, "Deeplab/AppleRustSet/OriginAug_label")

    if os.path.exists(origin_aug_label_folder):
        shutil.rmtree(origin_aug_label_folder)
    os.mkdir(origin_aug_label_folder)

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(origin_aug_label_folder):
        os.makedirs(origin_aug_label_folder)

    # 将 train(origin) 文件夹中的所有文件移动到 OriginAug 文件夹中
    for filename in os.listdir(origin_label_folder):
        source_path = os.path.join(origin_label_folder, filename)
        target_path = os.path.join(origin_aug_label_folder, filename)
        shutil.copy(source_path, target_path)

    # 将 State 文件夹中的所有文件复制到 OriginAug 文件夹中，并在文件名前添加 "aug" 字符串
    for filename in os.listdir(state_label_folder):
        source_path = os.path.join(state_label_folder, filename)
        target_filename = "aug" + filename
        target_path = os.path.join(origin_aug_label_folder, target_filename)
        shutil.copy(source_path, target_path)


    lr=0.001
    load_weight="./Deeplab/deeplab_weights/test_weights/deeplab_weights(batch32_lr0.001_Adam(rust)).pth"
    # load_weight = None
    save_weight="./Deeplab/deeplab_weights/test_weights"
    zone_name = "batch{0}_lr{1}_Adam".format(batch_size, lr)
    opti_type = 'Adam'
    tensorboard_path = './Deeplab/results/'
    trainset_path = ['Deeplab/AppleRustSet/OriginAug', 'Deeplab/AppleRustSet/OriginAug_label']
    CLASS_NUM = 2
    MOMENTUM = 0.9
    CATE_WEIGHT = [0.7502381287857225, 1.4990483912788268]

    deeplab = DeepLab(num_classes=CLASS_NUM,
                      backbone='mobilenet',
                      output_stride=16,
                      sync_bn=False,
                      freeze_bn=False)
    deeplab = deeplab.train()
    deeplab = deeplab.cuda()
    deeplab.load_state_dict(torch.load(load_weight))
    loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()

    optimizer = torch.optim.Adam(deeplab.parameters(), lr=lr, betas=(0.9, 0.99))

    train_data = Dataset(trainset_path[0], trainset_path[1])
    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    iter_count = 0
    Mean_IOU = [] # 后几个迭代的iou和集合

    for epoch in range(epochs):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = deeplab(b_x)
            loss = loss_func(output, b_y.long())
            loss = loss.cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5、迭代一次，打印一次，记录一次损失值
            # print("Epoch:{0} || Step:{1} || Loss:{2}".format(epoch, step, format(loss, ".6f")))
            iter_count = iter_count + 1
        # 完成一个epoch，保存一次权重用于算iou等指标，并保存；下个epoch会覆盖掉
        weight_file_name = "./Deeplab/deeplab_weights/test_weights/deeplab_weights(retrain).pth"
        torch.save(deeplab.state_dict(), weight_file_name)
    # weight_file_name = save_weight + "/deeplab_weights(retrain).pth"
    # torch.save(deeplab.state_dict(), weight_file_name)  # time.time()：1970纪元后经过的浮点秒数 # strftime:当前时间





        # 进入最后五个epoch，每一次都要计算模型结果
        if epoch >= epochs-5:
        # if True:
            # 参数利用微调的模型结果计算
            CLASS_NUM = 2
            Test_Weights = weight_file_name  # 训练好的权重路径
            deeplab_test = DeepLab(num_classes=CLASS_NUM,
                              backbone='mobilenet',
                              output_stride=16,
                              sync_bn=False,
                              freeze_bn=False)
            deeplab_test.load_state_dict(torch.load(Test_Weights))
            deeplab_test.eval()
            # cpu-cuda
            deeplab_test.cuda()

            IOU = []

            testset_path = ['Deeplab/AppleRustSet/Test(origin)', 'Deeplab/AppleRustSet/Test_label(origin)']
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

                # =======================用SegmentIndex方法来计算miou及pa等指标=======================================
                metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几
                hist = metric.addBatch(np.array(predict, dtype=np.uint8), np.array(label, dtype=np.uint8))
                iou = metric.IntersectionOverUnion()[1]
                IOU.append(iou)
                # print("iou_{0}:{1}".format(index, format(IOU[index], ".6f")))
            # 将各个集合中nan转为0
            IOU = np.nan_to_num(IOU)
            # print("average_iou:{}".format(format(np.mean(IOU), ".6f")))
            # 测试集的平均iou
            Mean_IOU.append(np.mean(IOU))
    return np.mean(Mean_IOU)


if __name__=='__main__':
    ReTrain_Deeplab(epochs=60, batch_size=4)

