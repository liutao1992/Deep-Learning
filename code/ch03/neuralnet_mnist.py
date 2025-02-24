# coding: utf-8
import sys, os
sys.path.append(os.pardir)            # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle                         # 用于加载保存的网络参数
from dataset.mnist import load_mnist  # 导入MNIST数据集
from common.functions import sigmoid, softmax  # 导入激活函数


def get_data():
    """获取MNIST数据集
    
    Returns:
        (x_test, t_test): 测试图像和测试标签
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True,      # 将图像的像素值正规化为0.0~1.0
        flatten=True,        # 将图像展开为一维数组
        one_hot_label=False  # 标签为0~9的数字，而不是one-hot编码
    )
    return x_test, t_test


def init_network():
    """
    加载预训练的网络参数    
    Returns:
        network: 包含权重和偏置的字典，结构如下：
        {
            'W1': array([784, 50]),   # 第一层权重：输入层(784节点)到第一隐藏层(50节点)
            'b1': array([50]),        # 第一层偏置
            'W2': array([50, 100]),   # 第二层权重：第一隐藏层(50节点)到第二隐藏层(100节点)
            'b2': array([100]),       # 第二层偏置
            'W3': array([100, 10]),   # 第三层权重：第二隐藏层(100节点)到输出层(10节点)
            'b3': array([10])         # 第三层偏置
        }
    """
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)  # 加载预训练的网络参数，包含上述权重和偏置
    return network


def predict(network, x):
    """进行预测的前向传播
    
    Args:
        network: 神经网络参数（权重和偏置）
        x: 输入数据
    
    Returns:
        y: 输出结果（概率分布）
    """
    # 获取网络参数
    W1, W2, W3 = network['W1'], network['W2'], network['W3']  # 权重参数
    b1, b2, b3 = network['b1'], network['b2'], network['b3']  # 偏置参数

    # 让我详细解释这行代码：
    # ```python
    # a1 = np.dot(x, W1) + b1  # 加权和
    # ```

    # 这是神经网络中的一个关键计算步骤，实现了输入层到第一隐藏层的变换：

    # ### 1. 组成部分
    # - `x`：输入数据（一张图片，784个像素值）
    # - `W1`：第一层权重矩阵（784×50）
    # - `b1`：第一层偏置（50个节点）
    # - `np.dot()`：NumPy的矩阵乘法函数

    # ### 2. 计算过程
    # 1. `np.dot(x, W1)`：
    # - x的形状：(784,)
    # - W1的形状：(784, 50)
    # - 结果形状：(50,)
    # - 每个结果值代表一个神经元的加权输入

    # 2. `+ b1`：
    # - 给每个神经元加上偏置值
    # - 使神经元的激活阈值可调节

    # ### 3. 具体例子
    # 假设简化版：
    # ```
    # x = [0.1, 0.2]          # 输入值（2个像素）
    # W1 = [[0.3, 0.5],      # 权重矩阵（2×2）
    #     [0.4, 0.6]]
    # b1 = [0.1, 0.2]        # 偏置值

    # 计算过程：
    # 第一个神经元的值 = (0.1×0.3 + 0.2×0.4) + 0.1
    # 第二个神经元的值 = (0.1×0.5 + 0.2×0.6) + 0.2
    # ```
    
    # 这个计算实现了神经网络中的"加权求和"操作，是神经网络处理信息的基本步骤。

    a1 = np.dot(x, W1) + b1  # 加权和

    z1 = sigmoid(a1)         # 激活函数
    
    # 第二层
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    # 输出层
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)  # 输出层使用softmax函数

    return y


# 获取测试数据
x, t = get_data()
# 加载网络参数
network = init_network()
# 用于记录识别正确的个数
accuracy_cnt = 0

# 对测试数据逐个进行预测
for i in range(len(x)):
    y = predict(network, x[i])  # 获取预测结果
    p = np.argmax(y)  # 获取概率最高的元素的索引
    if p == t[i]:     # 如果预测值与真实值相同
        accuracy_cnt += 1  # 正确计数加1

# 输出准确率
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))