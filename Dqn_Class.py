import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


# 2. Define the network used in both target net and the net for training
class DqnNet(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        # Define the network structure, a very simple fully connected network
        super(DqnNet, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc1
        self.fc2 = nn.Linear(10, 10)
        self.fc2.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc1
        self.out = nn.Linear(10, N_ACTIONS)  # layer 2
        self.out.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc2

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# 3. Define the DQN network and its corresponding methods
class DQN(object):

    # 定义训练相关参数和函数
    def __init__(self, MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR, EPSILON, BATCH_SIZE, GAMMA, TARGET_NETWORK_REPLACE_FREQ, DqnNet):
        self.memory_capacity = MEMORY_CAPACITY
        self.n_states = N_STATES
        self.n_actions = N_ACTIONS
        self.learning_rate = LR
        self.epsilon = EPSILON
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.target_nerwork_replace_freq = TARGET_NETWORK_REPLACE_FREQ
        self.eval_net, self.target_net = DqnNet, DqnNet # -----------Define 2 networks (target and training)------#
        self.loss = 0


        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # ----Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((self.memory_capacity, self.n_states * 2 + 2))

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy


        x = torch.unsqueeze(x, 0)  # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = int(action)
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, self.n_actions)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        a = a.reshape((-1, 1))
        r = r.reshape((-1, 1))
        transition = np.hstack((s, a, r, s_))  # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % self.memory_capacity
        self.memory[index:index+350, :] = transition
        self.memory_counter += 350

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.
        # 定义整个DQN的工作方式，包括采样批体验、何时以及如何更新目标网络的参数，以及如何实现反向传播。

        # update the target network every fixed steps
        # 每隔固定步骤更新目标网络
        if self.learn_step_counter % self.target_nerwork_replace_freq == 0:
            # Assign the parameters of eval_net to target_net
            # 将eval_net的参数分配给target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        # 确定缓冲batch中采样批次的索引
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        # 从批处理存储器中提取向量或矩阵s、a、r、s_，并将其转换为tensor变量
        # 便于反向传播
        b_s = Variable(torch.FloatTensor(b_memory[:, : self.n_states])).cuda()
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states: self.n_states + 1].astype(int))).cuda()
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1 : self.n_states + 2])).cuda()
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states :])).cuda()

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # print(q_eval)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate#与计算图分离，不要反向传播
        # select the maximum q value
        # print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # (batch_size, 1)
        self.loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        self.loss.backward()
        self.optimizer.step()  # execute back propagation for one step

