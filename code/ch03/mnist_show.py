# coding: utf-8
import sys, os
sys.path.append(os.pardir)            # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist  # 导入MNIST数据集加载函数
import matplotlib.pyplot as plt       # 导入绘图库

def img_show(img):
    """
    显示图像的函数
    参数:
        img: 输入的图像数组
    """
    plt.imshow(img, cmap='gray')  # 使用灰度模式显示图像
    plt.axis('off')               # 关闭坐标轴显示
    plt.show()                    # 显示图像

# 加载MNIST数据集
# flatten=True表示将图像展开为一维数组
# normalize=False表示不将像素值正规化到0.0~1.0
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 获取训练集中的第一张图像
img = x_train[0]
# 获取对应的标签
label = t_train[0]
print(label)               # 打印标签值（5）

print(img.shape)           # 打印展开后的图像形状(784,)
img = img.reshape(28, 28)  # 将图像重新整形为28x28的二维数组
print(img.shape)           # 打印重整后的图像形状(28, 28)

# 显示图像
img_show(img)
