import matplotlib.pyplot as plt
import numpy as np


# 边训练边绘制损失曲线
def loss_ing_draw(train_loss):
    y_train_loss = train_loss  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
    # 去除顶部和右边框框
    # ax = plt.axes()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    #
    # plt.xlabel('iters')  # x轴标签
    # plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, '-r')
    plt.pause(0.01)


# 完成训练后绘制曲线
def loss_ed_draw(train_loss_path):
    with open(train_loss_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"
    y_train_loss = np.asfarray(data, float)  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.show()