import torch.nn.functional as F
import numpy as np
from pycm import *
from matplotlib import pyplot as plt
from augmentation import *
from loss_func import *

# 记录参数并上传wandb
from utils.metric import AverageMeter, accuracy
import wandb
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef

def train(model, device, train_loader, val_loader, optimizer, lossfn_dict, epoch, epochs, alpha=1., beta=1.):
    """
    对模型进行一轮训练，并打印和上传相关的训练指标
    训练指标包括：训练标签位于模型输出前1的正确率，训练标签位于模型输出前2的正确率，训练的损失值
    :param beta: scl_loss前的系数
    :param alpha: ce_loss前的系数
    :param val_loader: 验证集
    :param lossfn_dict: 训练损失函数(字典索引)
    :param optimizer: 训练优化器(字典索引)
    :param train_loader: 训练集(data当中包含原始数据, 还有两个数据增强后的数据)
    :param model: 网络模型,包含backbone和classifier以及转换特征的mlp
    :param device: 训练使用的设备,cuda或cpu
    :param epoch: 训练轮数
    :param epochs: 训练总轮数
    :return: 训练标签位于模型输出前1的正确率
    """
    model.train()  # 模型为训练模式
    train_loss = AverageMeter()  # 统计训练损失
    train_scl_loss = AverageMeter()  # 统计对比损失
    train_ce_loss = AverageMeter()  # 统计对比损失
    train_top1 = AverageMeter()  # 统计训练top1准确率
    train_top2 = AverageMeter()  # 统计训练top5准确率

    for batch_idx, (data, target, data_transformed1, data_transformed2) in enumerate(train_loader):
        # 将训练集数据迁移到gpu上
        target, data, data_transformed1, data_transformed2 = target.to(device), \
                                                            data.to(device), \
                                                            data_transformed1.to(device), \
                                                            data_transformed2.to(device)

        feat_mlp2, feat_mlp3, logits = model(data, data_transformed1, data_transformed2, 'train')

        features = torch.cat([feat_mlp2.unsqueeze(1), feat_mlp3.unsqueeze(1)], dim=1)
        scl_loss = lossfn_dict['scl_loss'](features, target)
        ce_loss = lossfn_dict['ce_loss'](logits, target)

        loss = alpha * ce_loss + beta * scl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新损失,准确率
        with torch.no_grad():
            prec1, prec2 = accuracy(logits, target, topk=(1, 2))
            train_loss.update(loss.item(), n=target.size(0))
            train_ce_loss.update(ce_loss.item(), n=target.size(0))
            train_scl_loss.update(scl_loss.item(), n=target.size(0))
            train_top1.update(prec1.item(), n=target.size(0))
            train_top2.update(prec2.item(), n=target.size(0))

            # 判断loss当中是不是有元素非空，如果有就终止训练，并打印梯度爆炸
            if np.any(np.isnan(loss.item())):
                print("Gradient Explore")
                break
            # 每训练20个小批量样本就打印一次训练信息
            if batch_idx % 20 == 0:
                print('Epoch: [%d|%d] Step:[%d|%d], LOSS: %.5f' %
                    (epoch, epochs, batch_idx + 1, len(train_loader), loss.item()))
    wandb.log({
        "Train top 1 Acc": train_top1.avg,
        "Train top2 Acc": train_top2.avg,
        "Classifier Loss": train_ce_loss.avg,
        "Supcon Loss": train_scl_loss.avg}, commit=False)
    acc, f1, mcc = test(model, device, val_loader)
    return acc

def test(model, device, test_loader):
    """
    上传模型在测试集上的测试指标到wandb网页，
    测试指标包括：测试标签位于模型输出前1的正确率，测试标签位于模型输出前5的正确率，测试的损失值
    :param model: 网络模型(字典索引)
    :param device: 训练使用的设备，cuda或cpu
    :param test_loader: 测试训练集
    :return:
    """
    model.eval()
    test_loss = AverageMeter()
    test_top1 = AverageMeter()
    test_top2 = AverageMeter()
    test_target = torch.tensor([], device=device)
    test_output = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            logits = model(data=data, mode='eval')
            loss = F.cross_entropy(logits, target)

            test_target = torch.cat([test_target, target], dim=0)
            test_output.append(logits)

            prec1, prec2 = accuracy(logits, target, topk=(1, 2))

            test_top1.update(prec1.item(), n=target.size(0))
            test_top2.update(prec2.item(), n=target.size(0))
            test_loss.update(loss.item(), n=target.size(0))

    _, test_pred = torch.max(torch.vstack(test_output), dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(np.array(test_target.long().cpu()),
                                                            np.array(test_pred.long().cpu()),
                                                            average='macro', zero_division=0)
    mcc = matthews_corrcoef(np.array(test_target.long().cpu()), np.array(test_pred.long().cpu()))
    wandb.log({
        "Test top1 Acc": test_top1.avg,
        "Test top2 Acc": test_top2.avg,
        "Test Loss": test_loss.avg,
        "precision": precision,
        "recall": recall,
        "F1": f1})
    return test_top1.avg, f1, mcc


if __name__ == "__main__":
    pass
