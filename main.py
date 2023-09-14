import os
import random  # to set the python random seed
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 自定义的Dataloader,网络和损失函数
from loss_func import *
from trainer import train, test
from Model.CCQNet import CCQNet

# 记录参数并上传wandb
from utils.metric import *
from train_function import group_parameters
import wandb
import logging
from test import test_main

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

def main(config):
    # 配置训练模型时使用的设备(cpu/cuda)
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 如果使用cuda则修改线程数和允许加载数据到固定内存
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # 读取数据集
    file_name = config.datasets + '_' + 'IB' + str(config.IB_rate) + '_' + 'trainloader' + '.pth'
    train_loader = torch.load(r'/code/CCQNet/data/' + file_name)

    file_name = config.datasets + '_' + 'valloader' + '.pth'
    val_loader = torch.load(r'/code/CCQNet/data/' + file_name)

    # 设置随机数种子，保证结果的可复现性
    random.seed(config.seed)  # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    np.random.seed(config.seed)  # numpy random seed
    # 固定返回的卷积算法，保证结果的一致性
    torch.backends.cudnn.deterministic = True

    # 定义使用的网络模型（特征提取器和分类器）
    model = eval(config.model_names)(**config.model_params).to(device)

    # 定义损失函数
    lossfn_dict = {}
    lossfn_dict['scl_loss'] = eval(config.lossfn_names['scl_loss']) \
        (**config.lossfn_params['scl_loss']).to(device)
    lossfn_dict['ce_loss'] = eval(config.lossfn_names['ce_loss']) \
        (**config.lossfn_params['ce_loss']).to(device)

    # 定义训练模型的优化器
    group = group_parameters(model)
    optimizer = torch.optim.SGD([
        {"params": group[0], "lr": config.lr},  # weight_r
        {"params": group[1], "lr": config.lr * config.slr},  # weight_g
        {"params": group[2], "lr": config.lr * config.slr},  # weight_b
        {"params": group[3], "lr": config.lr},  # bias_r
        {"params": group[4], "lr": config.lr * config.slr},  # bias_g
        {"params": group[5], "lr": config.lr * config.slr},  # bias_b
        {"params": group[6], "lr": config.lr},
        {"params": group[7], "lr": config.lr},
    ], lr=config.lr, momentum=0.9, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                        eta_min=1e-8)  # goal: maximize Dice score

    # 追踪模型参数并上传wandb
    wandb.watch(model, log="all")

    # 定义存储模型参数的文件名
    model_paths = config.datasets + '_' + 'IB' + str(config.IB_rate) + '_' + 'CCQNet' + '.pth'

    # 训练模型
    max_acc = 0.
    for epoch in range(1, config.epochs + 1):
        acc = train(model=model, device=device, epoch=epoch, epochs=config.epochs, train_loader=train_loader,
            val_loader=val_loader, optimizer=optimizer, lossfn_dict=lossfn_dict,
            alpha=1., beta=1.)

        lr_scheduler.step()
        print('Epoch: %d, Test_acc: %.5f' % (epoch, acc))

        if epoch >= 100 and acc >= max_acc:
            max_acc = acc
            # 存储模型参数
            torch.save(model.state_dict(), model_paths)
            print("save model")
    test_main(config)


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

    main(config)
