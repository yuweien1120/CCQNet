import os
import random  # to set the python random seed
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 自定义的Dataloader
from processing import CWRU_DataLoader

# 记录参数并上传wandb
from utils.metric import *

def make_dataloader(datasets='CWRU'):
    # 读取数据集
    path = ''
    if datasets == 'CWRU':
        path = r'/root/autodl-tmp/0HP'
    elif datasets == 'HIT':
        path = r'/root/autodl-tmp/0.1HP-1800'
    IB = [5, 10, 20, 50]
    sample_trans_name = ("AddGaussian", "RandomScale",  'Randomstretch', "Randomcrop")
    for IB_rate in IB:
        train_loader, val_loader, test_loader = CWRU_DataLoader(d_path=path,
                                                                length=2048,
                                                                use_sliding_window=True,
                                                                train_number=500,
                                                                test_number=250,
                                                                valid_number=250,
                                                                batch_size=60,
                                                                normal=True,
                                                                IB_rate=IB_rate,
                                                                transforms_name=sample_trans_name,
                                                                seed=46)
        # 保存训练集和验证集
        file_name = datasets + '_' + 'IB' + str(IB_rate) + '_' + 'trainloader' + '.pth'
        torch.save(train_loader, r'data/' + file_name)
        # 保存验证集
        if IB_rate == 5:
            file_name = datasets + '_' + 'valloader' + '.pth'
            torch.save(val_loader, r'data/' + file_name)
    for seed in range(42, 51):
        train_loader, val_loader, test_loader = CWRU_DataLoader(d_path=path,
                                                                length=2048,
                                                                use_sliding_window=False,
                                                                train_number=500,
                                                                test_number=250,
                                                                valid_number=250,
                                                                batch_size=60,
                                                                normal=True,
                                                                IB_rate=5,
                                                                transforms_name=(),
                                                                seed=seed)
        # 保存测试集
        file_name = datasets + '_' + 'seed' + str(seed) + '_' + 'testloader' + '.pth'
        torch.save(test_loader, r'data/' + file_name)


if __name__ == '__main__':
    make_dataloader('HIT')
