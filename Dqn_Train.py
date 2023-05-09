from Dqn_Class import DQN, DqnNet
import torch
import os
import shutil
from ExecuteAction import do_img_aug, do_img_aug_label
import glob
import cv2 as cv
import numpy as np
from ReTrain_Deeplab import ReTrain_Deeplab
from ComputeCharacter import compute_character
from torch.utils.tensorboard import SummaryWriter
from time import strftime
from Deeplab_Test import Deeplab_Test
from Deeplab.Deeplab_Class import DeepLab
import json


BATCH_SIZE = 105  # batch size of sampling process from buffer
LR = 0.01  # learning rate
EPSILON = 0.9  # epsilon used for epsilon greedy approach
GAMMA = 0.9  # discount factor
TARGET_NETWORK_REPLACE_FREQ = 30  # How frequently target netowrk updates
MEMORY_CAPACITY = 2100  # The capacity of experience replay buffer

N_STATES = 12
N_ACTIONS = 8


deeplab_batch_sizes = 32
deeplab_epochs = 22

writer = SummaryWriter(log_dir='./results', filename_suffix=str(strftime('%Y%m%d_%H%M%S')))

dqnnet = DqnNet(N_STATES, N_ACTIONS).cuda()

dqn = DQN(MEMORY_CAPACITY=MEMORY_CAPACITY, N_STATES=N_STATES, N_ACTIONS=N_ACTIONS, LR=LR, EPSILON=EPSILON,
          BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TARGET_NETWORK_REPLACE_FREQ=TARGET_NETWORK_REPLACE_FREQ, DqnNet=dqnnet)

num_episodes = 500
max_steps_per_episode = 3

# 导入deeplab
deeplab_leaf = DeepLab(num_classes=2,
                  backbone='mobilenet',
                  output_stride=16,
                  sync_bn=False,
                  freeze_bn=False)
deeplab_leaf.load_state_dict(torch.load("./Deeplab/deeplab_weights/test_weights/deeplab_weights(batch64_lr0.001_Adam(leaf)).pth"))
deeplab_leaf.eval().cuda()

deeplab_rust = DeepLab(num_classes=2,
                  backbone='mobilenet',
                  output_stride=16,
                  sync_bn=False,
                  freeze_bn=False)
deeplab_rust.load_state_dict(torch.load("./Deeplab/deeplab_weights/test_weights/deeplab_weights(batch32_lr0.001_Adam(rust)).pth"))
deeplab_rust.eval().cuda()

# 计算r0
weight_file_name = "./Deeplab/deeplab_weights/test_weights/deeplab_weights(batch32_lr0.001_Adam(rust)).pth"
testset_path = ['Deeplab/AppleRustSet/Test(origin)', 'Deeplab/AppleRustSet/Test_label(origin)']
# Deeplab_Test(weight_file_name,"./train.txt")
# r0 = Deeplab_Test(weight_file_name, testset_path, save_path="Deeplab/AppleRustSet/leaf_Predict_label/")[3]
r0 = (0.828 + 0.83 + 0.8296 + 0.8322 + 0.8337)/5

count = 0
for episode in range(num_episodes):
    total_reward = 0
    r0 = (0.828 + 0.83 + 0.8296 + 0.8322 + 0.8337) / 5

    # 1、准备工作，清空，然后复制Train到State填充，并且创建State_label
    # 清空State文件夹
    state_dir = 'Deeplab/AppleRustSet/State'
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.mkdir(state_dir)
    # 复制Train(origin)下的文件到State文件夹
    train_dir = 'Deeplab/AppleRustSet/Train(origin)'
    for filename in os.listdir(train_dir):
        src_path = os.path.join(train_dir, filename)
        dst_path = os.path.join(state_dir, filename)
        shutil.copyfile(src_path, dst_path)

    # 清空State_label文件夹
    state_dir = 'Deeplab/AppleRustSet/State_label'
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.mkdir(state_dir)
    # 复制Train(origin)下的文件到State文件夹
    train_dir = 'Deeplab/AppleRustSet/Train_label(origin)'
    for filename in os.listdir(train_dir):
        src_path = os.path.join(train_dir, filename)
        dst_path = os.path.join(state_dir, filename)
        shutil.copyfile(src_path, dst_path)

    # 为下一个episode的第一步做准备工作，提前计算state
    paths = 'Deeplab/AppleRustSet/State/*'

    states = np.zeros((350, 12))
    actions = np.zeros(350, )
    states_ = np.zeros((350, 12))
    rewards = np.zeros(350, )

    paths = 'Deeplab/AppleRustSet/State/*'
    i = 0
    for path in glob.glob(paths):
        s1 = compute_character(path, deeplab_leaf)
        s2 = compute_character(path, deeplab_rust)
        states[i, :] = np.hstack([s1, s2])
        i = i+1

    # 2、进入step
    for step in range(max_steps_per_episode):
        # 1、定义环境：
        # s = torch.tensor(350, 12)
        # 路径：
        # dataset = EnvDataset('Deeplab/AppleRustSet/State')
        # Env_Batch_Size = 4
        # dataloader = DataLoader(dataset, batch_size=Env_Batch_Size, shuffle=True)
        # cnn = CNN().cuda().eval()
        #
        # for i, b_x in enumerate(dataloader, 0):
        #     img = b_x
        #     img = img.cuda().float()
        #     s = cnn(img)
        #     for i in range(Env_Batch_Size):
        #         state = s[i]
        #         a = dqn.choose_action(state)
        #         # do_img_aug(a)

        i = 0
        for path in glob.glob(paths):
            # path = path.replace('\\', '/')
            # img = cv.imread(path, flags=1)
            # img = cv.resize(img, (224, 224))
            # img = img / 255.0  # 归一化输入
            # img = torch.tensor(img).permute(2, 0, 1)
            # img = img.unsqueeze(0)
            # img = img.cuda().float()
            # s = cnn(img)
            a = dqn.choose_action(torch.tensor(states).cuda().float()[i])


            output_image = do_img_aug(path, a)
            cv.imwrite(path, output_image)
            output_label_image = do_img_aug_label(path.replace('State', 'State_label'), a)
            cv.imwrite(path.replace('State', 'State_label'), output_label_image)

            output_image = cv.resize(output_image, (224, 224))
            output_image = output_image / 255.0  # 归一化输入
            output_image = torch.tensor(output_image).permute(2, 0, 1)
            output_image = output_image.unsqueeze(0)
            output_image = output_image.cuda().float()

            s1_ = compute_character(path, deeplab_leaf)
            s2_ = compute_character(path, deeplab_rust)
            s1_ = torch.tensor(s1_)
            s2_ = torch.tensor(s2_)
            s_ = torch.cat((s1_, s2_), dim=0)
            s_ = s_.cuda().float()

            s_ = s_.detach().cpu().numpy().reshape(12, )
            states_[i, :] = s_
            actions[i] = a
            i = i + 1
        Last5epoch_Mean_IOU = ReTrain_Deeplab(epochs=deeplab_epochs, batch_size=deeplab_batch_sizes)
        if Last5epoch_Mean_IOU > r0:
            rewards.fill(100 * (Last5epoch_Mean_IOU - r0))
        else:
            rewards.fill(100 * (Last5epoch_Mean_IOU - r0))
        dqn.store_transition(states, actions, rewards, states_)

        if dqn.memory_counter >= MEMORY_CAPACITY:
            dqn.learn()
            # if done:
            #     print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))


        writer.add_scalar('Train/step_loss', dqn.loss, count)
        writer.add_scalar('Train/step_reward', rewards[0], count)
        # 保存动作结果到json中
        data = {
            'episode': episode,
            'step': step,
            'actions': actions.astype(int).tolist(),  # 将原numpy数组（float）转换新的（int）np数组，再转为列表
            'Last5epoch_Mean_IOU': Last5epoch_Mean_IOU
        }
        print(data)
        with open('actions.json', 'a') as f:
            json.dump(data, f)
            f.write('\n')

        total_reward += rewards[0]
        count += 1
        print('episode:{0} step:{1} loss:{2} rewards:{3}'.format(episode, step, dqn.loss, rewards[0]))
        states = states_
        r0 = Last5epoch_Mean_IOU

    writer.add_scalar('Train/episode_rewards', total_reward, episode)

    torch.save(dqnnet.state_dict(), "dqn_weights/dqn_weights.pth")
torch.save(dqnnet.state_dict(), "dqn_weights/dqn_weights(lr0.01_Adam_TARGET10).pth")

# # 1、记录时间
# starttime = datetime.datetime.now()
# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)