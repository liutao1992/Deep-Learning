# 让我用简单的方式解释这句话：MNIST的图像数据是28像素 ×28像素的灰度图像(1通道)，
# 各个像素 的取值在 0 到 255 之间

# ### 1. 图像尺寸：28×28
# - 就像一个28格×28格的方格纸
# - 总共有784个小格子(28×28=784)
# - 每个格子代表一个像素点

# ### 2. 灰度图像
# - 只有黑白和灰色，没有彩色
# - 每个像素点的值在0-255之间
#   - 0表示纯黑
#   - 255表示纯白
#   - 中间的数字表示不同程度的灰色

# ### 3. 1通道
# - 彩色图片通常有3个通道（RGB）
#   - R通道：控制红色
#   - G通道：控制绿色
#   - B通道：控制蓝色
# - 灰度图像只需要1个通道
#   - 因为只需要表示黑白程度
#   - 不需要表示颜色信息

# ### 4. 实际例子
# 想象手写数字"5"：
# ```
# □□■■■□□
# □■□□□■□
# □■□□□□□
# □□■■■□□
# □□□□□■□
# □■□□□■□
# □□■■■□□
# ```
# - 每个□或■代表一个像素
# - □表示接近白色（值接近255）
# - ■表示接近黑色（值接近0）

# 这就是为什么在代码中我们看到`img_dim = (1, 28, 28)`，表示1个通道，28行，28列。




# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip     # 用于解压缩gz文件
import pickle   # 用于序列化和反序列化Python对象
import os
import numpy as np

# MNIST数据集下载地址
url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'  # mirror site
# 定义数据集文件名
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',    # 训练图像
    'train_label':'train-labels-idx1-ubyte.gz',  # 训练标签
    'test_img':'t10k-images-idx3-ubyte.gz',      # 测试图像
    'test_label':'t10k-labels-idx1-ubyte.gz'     # 测试标签
}

# 获取数据集所在目录的绝对路径
dataset_dir = os.path.dirname(os.path.abspath(__file__))
# 保存处理后的数据集的文件路径
save_file = dataset_dir + "/mnist.pkl"  

# 定义数据集基本信息
train_num = 60000      # 训练数据数量
test_num = 10000       # 测试数据数量
img_dim = (1, 28, 28)  # 图像维度：通道数，高度，宽度
img_size = 784         # 展平后的图像大小：28x28=784

def _download(file_name):
    """下载单个数据集文件"""
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):  # 如果文件已存在则跳过
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    """下载所有MNIST数据集文件"""
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    """加载标签文件并转换为NumPy数组"""
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)  # 跳过文件头的8个字节
    print("Done")
    
    return labels

def _load_img(file_name):
    """加载图像文件并转换为NumPy数组"""
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)  # 跳过文件头的16个字节
    data = data.reshape(-1, img_size)  # 重塑为二维数组，每行是一张图片
    print("Done")
    
    return data
    
def _convert_numpy():
    """将所有数据转换为NumPy数组格式"""
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    """初始化MNIST数据集：下载并转换为NumPy数组，然后保存为pickle文件"""
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    """将标签转换为one-hot编码格式"""
    T = np.zeros((X.size, 10))  # 创建零矩阵
    for idx, row in enumerate(T):
        row[X[idx]] = 1  # 将对应位置设为1
        
    return T
    
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):  # 如果pickle文件不存在，则初始化数据集
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:  # 将像素值正规化到0.0~1.0
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:  # 转换为one-hot编码
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:  # 转换为卷积神经网络所需的形状
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 

# 如果直接运行此文件，则初始化数据集
if __name__ == '__main__':
    init_mnist()
