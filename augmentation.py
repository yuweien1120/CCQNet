import torch
import random
from torch.autograd import grad
from torchvision import transforms
import torch.nn as nn
import librosa
import numpy as np

# 以下的数据增强操作皆可微

def _covert_tensor_shape(x: torch.Tensor) -> torch.Tensor:
    """
    如果输入是一维张量，则将其转置为列向量（二维张量）
    :param x: 输入张量
    :return: 列向量
    """
    if x.ndim == 1:
        x = x.reshape([len(x), 1])
    return x


# 可微分的数据增强父类
class Augmentation:
    def __init__(self, p=0.5, temperature=1.):
        """
        初始化可微分的数据增强的参数
        :param p: 使用数据增强的概率(float or shape[batch_size])
        :param temperature: 松弛伯努利分布的温度系数(float or shape[1])
        """
        super().__init__()
        if torch.is_tensor(p):
            self.device = p.device  # 获取所有张量使用的设备
        else:
            self.device = 'cpu'
        self.p = p
        self.temperature = temperature

    def __setattr__(self, name, value):
        if name == 'temperature':
            temperature = torch.as_tensor(value, device=self.device)
            # 温度系数只使用一个
            if len(temperature.shape) == 0:
                temperature = temperature.reshape(1)
            elif len(temperature.shape) > 1:
                raise ValueError('temperature must only contain 1 dim or be a scalar.')
            elif temperature.shape[0] > 1:
                raise ValueError('temperature must only contain 1 element.')
            self.__dict__[name] = temperature
        elif name == 'device':
            self.__dict__[name] = value
        else:
            # 给数据增强的成员变量赋值时，需要先转换成列向量的形式(利用广播机制)
            value = torch.as_tensor(value, device=self.device)
            if len(value.shape) == 0:
                value = value.reshape(1)
            elif len(value.shape) > 1:
                raise ValueError(name + ' must only contain 1 dim.')
            value = _covert_tensor_shape(value)
            self.__dict__[name] = value

    def __call__(self, x):
        # 获取数据的batch_size, data_length
        row, col = x.shape
        # 建立松弛化的伯努利分布
        # 如果p只有一个元素,则将其重复样本数次,保证后面采样的值各个样本之间独立
        if self.p.shape[0] == 1:
            p = self.p.repeat(row, 1)
        elif self.p.shape[0] != row:
            raise ValueError('the number of batch_size must be the same as the number of elements in the p.')
        else:
            p = self.p
        p_dist = torch.distributions.RelaxedBernoulli(self.temperature, p)
        # 重参数化采样,该采样方式可以使得概率可导
        p_sample = p_dist.rsample()
        # 如果采样的值大于0.5,则使用数据增强
        p_sample_index = p_sample >= 0.5
        # 将采样的数值定为1或0(用于前向传播),然后存储松弛化伯努利分布采样的梯度值
        p_sample = p_sample_index.float() - p_sample.detach() + p_sample
        # 返回采样值
        return p_sample


class AddGaussian(Augmentation):
    def __init__(self, p=0.5, mean=0, u=0.01, temperature=1.):
        """
        初始化高斯噪声相关参数
        :param p: 使用数据增强的概率(float or shape[batch_size])
        :param mean: 均值(float or shape[batch_size])
        :param u: 标准差(float or shape[batch_size])
        :param temperature: 松弛化伯努利分布的温度系数(float or shape[1])
        """
        super().__init__(p, temperature)
        # 将输入的均值和标准差转化为列向量，后面可以利用广播机制
        self.mean = mean
        self.u = u

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入原始数据(shape[batch_size, data_length])
        :return: 增强后的数据(shape[batch_size, data_length])
        """
        # 生成采样值
        p_sample = super().__call__(x)
        row, col = x.shape
        # 产生高斯噪声(重参数化)
        noise = torch.randn(row, col, device=self.device) * self.u + self.mean
        noise = noise.to(self.device)
        aug_x = x + noise
        # 按采样值选择是否执行数据增强
        x = p_sample * aug_x + (1 - p_sample) * x
        return x


class RandomScale(Augmentation):
    # 随机信号缩放
    def __init__(self, p=0.5, u=0.01, temperature=1.):
        super().__init__(p, temperature)
        self.u = u

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_sample = super().__call__(x)
        row, col = x.shape
        # 生成随机缩放的增益
        scale = torch.randn(row, col, device=self.device) * self.u + 1
        scale = scale.to(self.device)
        aug_x = x * scale
        x = p_sample * aug_x + (1 - p_sample) * x
        return x


class MaskNoise(Augmentation):
    # 添加掩码噪声
    def __init__(self, p=0.5, u=0.1, temperature=1.):
        super().__init__(p, temperature)
        self.u = u

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_sample = super().__call__(x)
        row, col = x.shape
        # 生成掩码矩阵
        if self.u.shape[0] == 1:
            u = self.u.repeat(row, col)
        elif self.u.shape[0] != row:
            raise ValueError('the number of batch_size must be the same as the number of elements in the u.')
        else:
            u = self.u.repeat(1, col)
        u_dist = torch.distributions.RelaxedBernoulli(self.temperature, u)
        u_sample = u_dist.rsample()
        Mask = u_sample.le(0.5).float()
        Mask = Mask - u_sample.detach() + u_sample
        Mask = Mask.to(self.device)
        aug_x = Mask * x
        x = p_sample * aug_x + (1 - p_sample) * x
        return x

class Translation(Augmentation):
    # 随机将信号某一位置两边的序列调换位置
    def __init__(self, p=0.5, temperature=1.):
        super().__init__(p, temperature)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_sample = super().__call__(x)
        row, col = x.shape
        # 随机位置交换序列
        aug_x = torch.zeros_like(x, device=self.device)
        for i in range(row):
            j = random.randint(0, col - 1)
            aug_x[i] = torch.roll(x[i], shifts=j, dims=0)
        x = p_sample * aug_x + (1 - p_sample) * x
        return x

class Randomstretch(Augmentation):
    # 随机拉伸信号
    def __init__(self, p=0.5, temperature=1.):
        super().__init__(p, temperature)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_sample = super().__call__(x)
        row, col = x.shape
        aug_x = torch.zeros_like(x, device=self.device)
        for i in range(row):
            rate = random.uniform(0, 1)
            temp = librosa.effects.time_stretch(np.asarray(x[i]), rate=rate)  # 过采样然后截断
            aug_x[i] = torch.tensor(temp[0:col])
        x = p_sample * aug_x + (1 - p_sample) * x
        return x

class Randomcrop(Augmentation):
    def __init__(self, p=0.5, segment_length=30, temperature=1.):
        super().__init__(p, temperature)
        self.segment_length = segment_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_sample = super().__call__(x)
        row, col = x.shape
        aug_x = torch.zeros_like(x, device=self.device)
        for i in range(row):
            j = random.randint(0, col - self.segment_length)
            x[i, j:j + self.segment_length] = 0
            aug_x[i] = x[i]
        x = p_sample * aug_x + (1 - p_sample) * x
        return x


def data_transform(data, transforms_params_dict):
    """
    将原始数据经过数据增强,返回增强后的数据
    :param data: 原始数据(shape[batch_size, data_length])
    :param transforms_params_dict: 不同数据增强的初始化参数组成的字典(dict)
    :return: 增强后的数据
    """
    transforms_list = []
    transforms_names = transforms_params_dict.keys()
    for name in transforms_names:
        transforms_list.append(eval(name)(**transforms_params_dict[name]))
    transform = transforms.Compose(transforms_list)
    transformed_data = transform(data)
    return transformed_data

def generate_params_dict(transforms_name, transforms_p=None, transforms_u=None, transforms_temperature=None):
    """
    生成数据增强参数组成的字典
    :param transforms_name: 数据增强的函数名(list or tuple)
    :param transforms_p: 数据增强执行的概率
    :param transforms_u: 数据增强的方差参数
    :param transforms_temperature: 松弛化的温度系数
    :return: 参数字典
    """
    transforms_params_dict = {}
    for i, name in enumerate(transforms_name):
        transforms_params_dict[name] = {}
        if not transforms_p is None:
            transforms_params_dict[name]['p'] = transforms_p[i]
        if not transforms_temperature is None:
            transforms_params_dict[name]['temperature'] = transforms_temperature
        if not transforms_u is None:
            if 'u' in eval(name).__init__.__code__.co_varnames:
                transforms_params_dict[name]['u'] = transforms_u[i]
    return transforms_params_dict

def params_index(index, params):
    """
    将参数张量按照索引张量重新生成参数张量
    :param index: 索引(list or tensor)
    :param params: 参数(tensor)
    :return: 新的参数张量(tensor)
    """
    if len(params.shape) == 1:
        temp = torch.zeros([len(index)], device=params.device)
        for i, j in enumerate(index):
            temp[i] = params[j]
        return torch.tensor([params[i] for i in index])
    elif len(params.shape) == 2:
        temp = torch.zeros([params.shape[0], len(index)], device=params.device)
        for i, param in enumerate(params):
            for j, k in enumerate(index):
                temp[i][j] = param[k]
        return temp
    else:
        raise ValueError("Invalid shape parameter: %s" % str(params.shape))

def data_augment(data, target, params, transforms_name, transforms_temperature):
    """
    根据样本类别施加数据增强
    :param data: 原始数据(shape[batch_size, data_length])
    :param target: 类别索引(list or tensor)
    :param params: 参数(tensor)
    :param transforms_name: 数据增强操作的名称(tuple or list)
    :param transforms_temperature: 松弛化的温度系数
    :return: 数据增强后的数据
    """
    py = params[2]  # 数据增强的概率参数
    uy = params[3]  # 数据增强的幅度参数
    py_index = params_index(target, py)  # 将概率参数按样本类别索引分配
    py_index = torch.sigmoid(6. * py_index)  # 将概率参数限制到[0, 1]区间上
    uy_index = params_index(target, uy)  # 将幅度参数按样本类别索引分配
    uy_index = torch.sigmoid(6. * uy_index)  # 将幅度参数限制到[0, 1]区间上
    # 生成存储数据增强概率和幅度参数(按照类别索引分配后)的字典
    transforms_params_dict = generate_params_dict(transforms_name, py_index, uy_index, transforms_temperature)
    aug_data = data_transform(data, transforms_params_dict)  # 对样本做数据增强
    return aug_data

if __name__ == '__main__':
    pass


