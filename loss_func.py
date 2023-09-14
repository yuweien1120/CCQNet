"""设计加入先验概率的损失函数"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class CRCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(CRCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, features, targets):
        """
        平衡的对比损失函数
        :param features: 由两组数据增强后的数据生成的feature(shape[batch_size, 2, feature_dim])
        :param targets: 特征标签(shape:[batch_size, 1])
        :return: 平衡对比损失
        """
        device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets = targets.repeat(2, 1)
        batch_cls_count = torch.eye(len(self.cls_num_list))[targets].sum(dim=0).squeeze()  # 统计各类别的样本数

        # mask矩阵(shape[2 * batch_size, 2 * batch_size])
        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        # logits_mask是对角元素为0，其他元素为1的矩阵
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        # 因为损失函数表达式的分子分母都不需要计算对自身的相似度
        # 对角元素置为0
        mask = mask * logits_mask

        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # 将features切片并组合成[2 * batch_size]
        logits = features.mm(features.T)  # 计算features之间相似度
        logits = torch.div(logits, self.temperature)

        # 数值稳定性
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 负例类别平衡
        exp_logits = torch.exp(logits) * logits_mask
        # 计算公式中的Wa
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)  # 对比损失的分母部分重加权

        log_prob = logits - torch.log(exp_logits_sum)  # 计算公式里的求和中的项
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 求平均

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss

class LC(nn.Module):
    def __init__(self, cls_num_list, tau=1., weight=None):
        super(LC, self).__init__()
        cls_num_list = torch.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, targets):
        """
        计算参数化的交叉熵损失函数
        :param x: 分类器输出的值(shape[batch_size, num_classes]))
        :param targets: 样本的标签(shape[batch_size])
        :return: 损失函数值(scalar)
        """
        device = (torch.device('cuda')
                if x.is_cuda
                else torch.device('cpu'))
        x_m = x + self.m_list.to(device)
        return F.cross_entropy(x_m, targets, weight=self.weight)


if __name__ == "__main__":
    pass
