import os
import random  # to set the python random seed
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 自定义的Dataloader,网络和损失函数
from loss_func import *
from trainer import train, test
from Model.QResNet import QResNet
from Model.CCQNet import CCQNet

# 记录参数并上传wandb
from utils.metric import *
from train_function import group_parameters
import wandb
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

class CCQNet1(nn.Module):
    def __init__(self, num_classes=10, dim_in=128, feat_dim=50):
        super(CCQNet1, self).__init__()
        self.encoder = QResNet()  # backbone
        self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
                                nn.Linear(dim_in, feat_dim))  # mlp模型转换backbone输出的特征
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, data, data_transformed1=None, data_transformed2=None):
        feat1 = self.encoder(data)
        return feat1

def test_main(config):
    # 配置训练模型时使用的设备(cpu/cuda)
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 如果使用cuda则修改线程数和允许加载数据到固定内存
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # 定义使用的网络模型（特征提取器和分类器）
    model = eval(config.model_names)(**config.model_params).to(device)
    model1 = CCQNet1(**config.model_params).to(device)

    # 加载模型参数
    # 定义存储模型参数的文件名
    model_paths = config.datasets + '_' + 'IB' + str(config.IB_rate) + '_' + 'CCQNet' + '.pth'
    model.load_state_dict(torch.load(model_paths))
    model1.load_state_dict(torch.load(model_paths))

    # 读取数据集
    test_acc = []
    test_f1 = []
    test_mcc = []

    for seed in range(42, 51):
        file_name = config.datasets + '_' + 'seed' + str(seed) + '_' + 'testloader' + '.pth'
        test_loader = torch.load(r'/code/CCQNet/data/' + file_name)

        if seed == 42:
            model.eval()
            model1.eval()
            test_target = torch.tensor([], device=device)
            test_output = []
            test_output1 = []

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data=data, mode='eval')
                    # feat_mlp2, feat_mlp3, logits, centers_logits = model(data=data, mode='train')
                    output1 = model1(data=data)
                    test_target = torch.cat([test_target, target], dim=0)
                    test_output.append(output)
                    test_output1.append(output1)

            _, test_pred = torch.max(torch.vstack(test_output), dim=1)
            tt_path = "test_target_" + config.datasets + '_' + 'IB' + str(config.IB_rate) + '_' + 'CCQNet' + '.pth'
            tp_path = "test_pred_" + config.datasets + '_' + 'IB' + str(config.IB_rate) + '_' + 'CCQNet' + '.pth'
            to1_path = "test_output1_" + config.datasets + '_' + 'IB' + str(config.IB_rate) + '_' + 'CCQNet' + '.pth'
            torch.save(test_target, tt_path)
            torch.save(test_pred, tp_path)
            torch.save(test_output1, to1_path)


        acc, f1, mcc = test(model, device, test_loader)
        test_acc.append(acc)
        test_f1.append(f1 * 100)
        test_mcc.append(mcc * 100)

    print("test_acc: mean: {:.2f}".format(np.mean(test_acc)), "std: {:.2f}".format(np.std(test_acc)))
    print("test_f1: mean: {:.2f}".format(np.mean(test_f1)), "std: {:.2f}".format(np.std(test_f1)))
    print("test_mcc: mean: {:.2f}".format(np.mean(test_mcc)), "std: {:.2f}".format(np.std(test_mcc)))
    wandb.log({
        "test_acc": np.mean(test_acc),
        "test_f1": np.mean(test_f1),
        "test_mcc": np.mean(test_mcc)})


if __name__ == '__main__':
    # 定义wandb上传项目名
    IB_rate = 50
    # name = 'IB= ' + str(IB_rate) + ':' + '1'
    name = 'all_IB_rate'
    wandb.init(project="CCQNet", name=name)
    wandb.watch_called = False

    # 定义上传的超参数
    config = wandb.config
    # 数据集及其预处理
    config.datasets = 'CWRU'  # 数据集
    # 采样时使用数据增强生成增强样本,
    # 不用每次训练都做一次增强, 节省运行时间
    config.sample_trans_name = ("AddGaussian", "RandomScale", 'Randomstretch', "Randomcrop")
    config.batch_size = 60  # 批量样本数
    config.IB_rate = IB_rate
    config.use_sampler = False  # 使用均匀采样器(不使用)
    config.test_batch_size = 60  # 测试批量样本数
    config.length = 2048
    config.train_number = 500  # 每一类的训练样本数
    config.test_number = 250  # 每一类的测试样本数

    config.num_classes = 10  # 样本类别数
    config.normal_index = 9  # CWRU/HIT正常样本的类别号

    # 网络模型的参数
    config.model_names = 'CCQNet'  # 网络模型
    config.model_params = {'num_classes': config.num_classes, 'dim_in': 128, 'feat_dim': 50}

    # 损失函数的参数
    config.lossfn_names = {'scl_loss': 'CRCL',
                        'ce_loss': 'LC'}  # 损失函数
    config.temperature = 0.1
    config.tau = 1.
    config.cls_num_list = config.num_classes * [config.train_number / IB_rate]  # 每一类的样本个数
    config.cls_num_list[config.normal_index] = config.train_number  # 正常类样本个数为0
    CRCL_params = {'cls_num_list': config.cls_num_list, 'temperature': config.temperature}  # 对比损失函数的初始化参数
    LC_params = {'cls_num_list': config.cls_num_list, 'tau': config.tau}
    config.lossfn_params = {'scl_loss': CRCL_params,
                            'ce_loss': LC_params}  # 损失函数的参数

    # 训练的相关参数
    config.epochs = 200  # 训练轮数
    config.optimizer = 'SGD'
    config.lr = 0.05
    config.slr = 0.00001

    config.weight_decay = 0.0005
    config.momentum = 0.9

    config.no_cuda = False  # 不使用cuda(T/F)

    config.seed = 46

    test_main(config)
