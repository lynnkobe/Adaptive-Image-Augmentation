import cv2 as cv
import numpy as np
import torch.utils.data as Data
import argparse
from time import strftime
from torch.utils.tensorboard import SummaryWriter
from Deeplab.Deeplab_Class import *
from Deeplab.SegmentIndex import SegmentationMetric
# 记录程序运行时间
import datetime
import time
from Deeplab.DeeplabDataset import Dataset
from Deeplab_Test import Deeplab_Test


# 网络训练
def Deeplab_Train(epoch, batch_size, lr, load_weight, save_weight, zone_name, opti_type,tensorboard_path, trainset_path, testset_path):
    global IOU

    # 相关参数载入
    CLASS_NUM = 2
    EPOCH = epoch
    BATCH_SIZE = batch_size
    LR = lr
    MOMENTUM = 0.9
    CATE_WEIGHT = [0.7502381287857225, 1.4990483912788268]  # 损失函数中类别的权重
    PRE_TRAINING = load_weight  # 网络参数加载路径
    WEIGHTS = save_weight  # 网络参数保存路径

    # 1、创建网络
    deeplab = DeepLab(num_classes=CLASS_NUM,
                    backbone='mobilenet',
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=False)
    deeplab = deeplab.train()
    deeplab = deeplab.cuda()

    # 2、网络参数初始化
    # deeplab.load_weights(PRE_TRAINING)
    if PRE_TRAINING != None:
        deeplab.load_state_dict(torch.load(PRE_TRAINING))
    # 优化器创建
    if opti_type == 'Adam':
        optimizer = torch.optim.Adam(deeplab.parameters(), lr=LR, betas=(0.9, 0.99)) # 比没betas效果好
    elif opti_type == 'SGD':
        optimizer = torch.optim.SGD(deeplab.parameters(), lr=LR)
    elif opti_type == 'SGD_MOMENTUM':
        optimizer = torch.optim.SGD(deeplab.parameters(), lr=LR, momentum=MOMENTUM)
    elif opti_type == 'RMSprop_alpha':
        optimizer = torch.optim.RMSprop(deeplab.parameters(), lr=LR, alpha=0.9)
    elif opti_type == 'Adam_betas':
        optimizer = torch.optim.Adam(deeplab.parameters(), lr=LR, betas=(0.9, 0.99))
    # 损失函数
    loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(CATE_WEIGHT)).float()).cuda()

    # 3、数据集创建
    train_data = Dataset(trainset_path[0], trainset_path[1])
    train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 4、Tensorboard创建(训练过程数据可视化)
    writer = SummaryWriter(log_dir=tensorboard_path + zone_name, filename_suffix=str(strftime('%Y%m%d_%H%M%S')))  # comment='test_your_comment', filename_suffix="_test_your_filename_suffix"
    iter_count = 0

    for epoch in range(EPOCH):
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
            print("Epoch:{0} || Step:{1} || Loss:{2}".format(epoch, step, format(loss, ".6f")))
            writer.add_scalar('Train/Loss', loss.item(), iter_count)
            iter_count = iter_count + 1

        # 5、完成一个epoch，保存一次权重用于算iou等指标，并保存；下个epoch会覆盖掉
        weight_file_name = WEIGHTS + "/deeplab_weights({}).pth".format(zone_name)
        torch.save(deeplab.state_dict(), weight_file_name)  # time.time()：1970纪元后经过的浮点秒数 # strftime:当前时间
        PA, CPA, MPA, IOU, MIOU = Deeplab_Test(weight_file_name=weight_file_name, testset_path=testset_path)
        print("Epoch:{0} || PA:{1} CPA:{2} MPA:{3} IOU:{4} mIOU:{5}".format(epoch, float(format(PA, '.6f')), float(format(CPA, '.6f')),
                                                                            float(format(MPA, '.6f')),float(format(IOU, '.6f')),float(format(MIOU, '.6f'))))
        writer.add_scalar('Train/PA', float(format(PA, '.6f')), epoch)
        writer.add_scalar('Train/CPA', float(format(CPA, '.6f')), epoch)
        writer.add_scalar('Train/MPA', float(format(MPA, '.6f')), epoch)
        writer.add_scalar('Train/IOU', float(format(IOU, '.6f')), epoch)
        writer.add_scalar('Train/MIOU', float(format(MIOU, '.6f')), epoch)
        writer.add_scalars('Train/evaluate', {'PA': float(format(PA, '.6f')), 'CPA': float(format(CPA, '.6f')), 'MPA': float(format(MPA, '.6f')),
                                              'IOU': float(format(IOU, '.6f')), 'MIOU': float(format(MIOU, '.6f'))}, epoch)

    # 6、保存训练结束后的网络参数。记录时间
    weight_file_name = WEIGHTS +"/deeplab_weights({})".format(zone_name) + str(strftime('%Y%m%d_%H%M%S')) + ".pth"
    torch.save(deeplab.state_dict(), weight_file_name)  # time.time()：1970纪元后经过的浮点秒数 # strftime:当前时间

    return IOU


# ————————————************调参修改：512——224，batch=32,lr=0.001,Adam，SummaryWriter路径名,是否载入预训练模型********************————————————————————————————
epoch=1
batch_size=4
lr=0.001
load_weight="./Deeplab/deeplab_weights/test_weights/deeplab_weights(batch32_lr0.001_Adam(rust)).pth"
# load_weight = None
save_weight="./Deeplab/deeplab_weights/test_weights"
zone_name = "batch{0}_lr{1}_Adam".format(batch_size, lr)
opti_type = 'Adam'
tensorboard_path = './Deeplab/results/'
trainset_path = ['Deeplab/AppleRustSet/Train(origin)', 'Deeplab/AppleRustSet/Train_label(origin)']
testset_path = ['Deeplab/AppleRustSet/Test(origin)', 'Deeplab/AppleRustSet/Test_label(origin)']

# 1、记录开始时间
starttime = datetime.datetime.now()
# 停留2秒
time.sleep(2)

# BN层的momentum
# bn_momentum = 0.1
# 设置随机种子
torch.cuda.manual_seed(1)
# 2、设置相关路径


# 3、进入训练
Deeplab_Train(epoch=epoch, batch_size=batch_size, lr=lr, load_weight=load_weight, save_weight=save_weight, zone_name=zone_name,
              opti_type=opti_type, tensorboard_path=tensorboard_path, trainset_path=trainset_path, testset_path=testset_path)

# 4、记录结束时间
endtime = datetime.datetime.now()
# 打印
print((endtime - starttime).seconds)

# if __name__=='__main__':
# if 0:
    # 调参改动：
    # ① 更改及决定是否导入上此训练的权重参数
    # weight_file_name_last = "./deeplab_weights/deeplab_weights(batch=12,lr=0.00005,Adam,512,512).pth"
    # deeplab.load_state_dict(torch.load(PRE_TRAINING)
    # ② 更改训练参数
    # parser.add_argument("--batch_size", type=int, default=4, help="批训练大小")
    # parser.add_argument("--learning_rate", type=float, default=0.00005, help="学习率大小")
    # ③ 更改保存tensorboard路径
    # writer = SummaryWriter(log_dir='/root/tf-logs/deeplab(batch=12,lr=0.00005,Adam,512,512)')
    # ④ 更改图片像素是否resize：(-1, 512, 512) / 255
    # ⑤ 更改参数更新方式：torch.optim.Adam(deeplab.parameters(), lr=LR, betas=(0.9, 0.99))
    # ⑥ 更改参数保存的路径
    # weight_file_name = "deeplab_weights/deeplab_weights" + str(strftime('%Y%m%d_%H%M%S')) + "(batch=12,lr=0.00005,Adam,512,512).pth"
    # ⑦ 若更改了样本集合，则改下面生成txt的路径：make_test_txt，make_train_txt
    # weight_file_name_last = "./deeplab_weights/deeplab_weights20221207_123829(batch=12,lr=0.00001,Adam,512,512).pth"
    # weight_file_name = train_weights_loss(weight_file_name_last, 100, 1000)
    # # 记录结束时间
    # endtime = datetime.datetime.now()
    # # 打印
    # print((endtime - starttime).seconds)

# if __name__=='__main__':
# # if 0:
#     weight_file_name="./test_weights/deeplab_weights0.934577.pth"
#     test_IMOU(weight_file_name)
