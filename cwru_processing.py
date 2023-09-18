from scipy.io import loadmat
import os
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from scipy.fft import fft
import random

from augmentation import *

def CWRU_DataLoader(d_path, length=2048, use_sliding_window=True,
                    train_number=1800,
                    test_number=300,
                    valid_number=300,
                    batch_size=64,
                    normal=True,
                    IB_rate=10,
                    transforms_name=("AddGaussian", "RandomScale", "MaskNoise", "Translation"),
                    use_fft=False,
                    seed=46):
    """对数据进行预处理,返回train_X, train_Y, test_X, test_Y样本.
    :param use_fft: 使用fft预处理原始数据
    :param use_sliding_window: 用滑窗采样/随机取初始点采样(默认True,滑窗采样)
    :param d_path: 源数据地址
    :param length: 信号长度，默认400
    :param train_number: 正常样本数(默认1800)
    :param test_number: 测试集的样本数
    :param valid_number: 验证集的样本数
    :param normal: 是否标准化.True,False.默认True
    :param batch_size: 批量大小
    :param IB_rate: 训练集中正常样本和每一个故障样本数的比例
    :param transforms_name: 生成数据增强的方法默认("Add_Gaussian", "Random_Scale", "Mask_Noise", "Translation")
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    """

    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    # 将文件名列表中结尾不是.mat的文件名去掉，防止下面loadmat报错
    for i in filenames:
        if not i.endswith('.mat'):
            filenames.remove(i)

    def capture():
        """读取mat文件，返回字典
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    def slice(data, train_number=train_number, test_number=test_number, valid_number=valid_number):
        """
        将数据按数据集的样本数提取
        :param data: 单挑数据
        :param train_number: 训练样本数
        :param test_number: 测试集的样本数
        :param valid_number: 验证集的样本数
        :param val_unbal: 验证集不平衡采样(默认False)
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Valid_Samples = {}
        Test_Samples = {}

        for i in keys:
            slice_data = data[i]
            if "Normal" in i or "normal" in i:  # 如果是正常样本，则正常采样
                samp_train = int(train_number)  # 500
            else:  # 如果是故障样本，则按IB_rate的比例采样
                samp_train = int(train_number / IB_rate)
            samp_valid = int(valid_number)
            samp_test = int(test_number)

            # 存储采集样本
            Train_Sample = []
            Valid_Sample = []
            Test_Sample = []
            start = 0  # 记录每次滑窗采集的初始位置

            all_length = len(slice_data)

            # 计算采样区间的左右索引
            train_start = 0
            train_end = int(all_length / 3)
            valid_start = train_end
            valid_end = int(all_length / 3) * 2
            test_start = valid_end
            test_end = int(all_length)

            np.random.seed(seed)
            # 抓取训练数据
            step_size = int((train_end - length) // samp_train)
            start = train_start
            for j in range(samp_train):
                if use_sliding_window:
                    sample = slice_data[start:start + length]
                    start = start + step_size
                else:
                    random_start = np.random.randint(low=train_start, high=(train_end - length))
                    sample = slice_data[random_start:random_start + length]
                if use_fft:
                    Train_Sample.append(np.abs(fft(sample)))
                else:
                    Train_Sample.append(sample)

            # 抓取验证数据
            step_size = int((train_end - length) // samp_valid)
            start = valid_start
            for h in range(samp_valid):
                if use_sliding_window:
                    sample = slice_data[start:start + length]
                    start = start + step_size
                else:
                    random_start = np.random.randint(low=valid_start, high=(valid_end - length))
                    sample = slice_data[random_start:random_start + length]
                if use_fft:
                    Valid_Sample.append(np.abs(fft(sample)))
                else:
                    Valid_Sample.append(sample)

            # 抓取测试数据
            step_size = int((train_end - length) // samp_test)
            start = test_start
            for h in range(samp_test):
                if use_sliding_window:
                    sample = slice_data[start:start + length]
                    start = start + step_size
                else:
                    random_start = np.random.randint(low=test_start, high=(test_end - length))
                    sample = slice_data[random_start:random_start + length]
                if use_fft:
                    Test_Sample.append(np.abs(fft(sample)))
                else:
                    Test_Sample.append(sample)
            Train_Samples[i] = Train_Sample
            Valid_Samples[i] = Valid_Sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Valid_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def scalar_stand(Train_X, Valid_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Valid_X = scalar.transform(Valid_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Valid_X, Test_X

    torch.manual_seed(seed)  # pytorch random seed(抽取数据集的随机数种子)
    np.random.seed(seed)  # numpy random seed
    random.seed(seed)  # python random seed
    # 从所有.mat文件中读取出数据的字典
    data = capture()
    # 将数据切分为训练集、测试集
    train, valid, test = slice(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为验证集制作标签，返回X，Y
    Valid_X, Valid_Y = add_labels(valid)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Valid_X, Test_X = scalar_stand(Train_X, Valid_X, Test_X)

    # 需要做一个数据转换，转换成tensor格式.
    # tensor是函数，可以生成指定类型的张量
    # Tensor是类，是默认张量类型torch.FloatTensor()的别名，生成float类型的张量
    Train_X = torch.tensor(Train_X, dtype=torch.float)
    Valid_X = torch.tensor(Valid_X, dtype=torch.float)
    Test_X = torch.tensor(Test_X, dtype=torch.float)
    Train_Y = torch.tensor(Train_Y, dtype=torch.long)
    Valid_Y = torch.tensor(Valid_Y, dtype=torch.long)
    Test_Y = torch.tensor(Test_Y, dtype=torch.long)

    Valid_X = Valid_X.reshape(Valid_X.shape[0], 1, Valid_X.shape[1])  # 增加通道维度
    Test_X = Test_X.reshape(Test_X.shape[0], 1, Test_X.shape[1])  # 增加通道维度

    if len(transforms_name) == 0:
        Train_X = Train_X.reshape(Train_X.shape[0], 1, Train_X.shape[1])  # 增加通道维度
        train_dataset = TensorDataset(Train_X, Train_Y)
    else:
        torch.manual_seed(seed)  # pytorch random seed(数据增强的随机数种子)
        np.random.seed(seed)  # numpy random seed
        random.seed(seed)  # python random seed
        # 生成数据增强的数据
        transforms_params_dict = generate_params_dict(transforms_name)
        Transformed_data1 = data_transform(Train_X, transforms_params_dict)
        Transformed_data2 = data_transform(Train_X, transforms_params_dict)

        # 增加通道维度
        Train_X = Train_X.reshape(Train_X.shape[0], 1, Train_X.shape[1])
        Transformed_data1 = Transformed_data1.reshape(Transformed_data1.shape[0], 1, Transformed_data1.shape[1])
        Transformed_data2 = Transformed_data2.reshape(Transformed_data2.shape[0], 1, Transformed_data2.shape[1])
        # 训练集包含了原样本和标签，还有两个经过数据增强后的样本，用于后面训练时计算对比损失
        train_dataset = TensorDataset(Train_X, Train_Y, Transformed_data1, Transformed_data2)

    valid_dataset = TensorDataset(Valid_X, Valid_Y)
    test_dataset = TensorDataset(Test_X, Test_Y)

    torch.manual_seed(seed)  # pytorch random seed(打乱数据集用的随机数种子)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    path = r'/root/autodl-tmp/0HP'
    transforms_name = ()  # 不使用数据增强
    train_loader, val_loader, test_loader = CWRU_DataLoader(d_path=path,
                                                            length=2048,
                                                            use_sliding_window=True,
                                                            train_number=500,
                                                            test_number=250,
                                                            valid_number=250,
                                                            batch_size=60,
                                                            normal=True,
                                                            IB_rate=50,
                                                            transforms_name=(),
                                                            seed=42)

