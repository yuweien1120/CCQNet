import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.QResNet import QResNet

class CCQNet(nn.Module):
    def __init__(self, num_classes=10,  dim_in=128, feat_dim=50):
        super(CCQNet, self).__init__()
        self.encoder = QResNet()  # backbone
        self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
                                nn.Linear(dim_in, feat_dim))  # mlp模型转换backbone输出的特征
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, data, data_transformed1=None, data_transformed2=None, mode='train'):
        if mode == 'train':
            feat1 = self.encoder(data)
            feat2 = self.encoder(data_transformed1)
            feat3 = self.encoder(data_transformed2)
            feat_mlp2 = F.normalize(self.head(feat2), dim=1)
            feat_mlp3 = F.normalize(self.head(feat3), dim=1)
            logits = self.fc(feat1)
            return feat_mlp2, feat_mlp3, logits
        elif mode == 'eval':
            feat = self.encoder(data)
            logits = self.fc(feat)
            return logits
        else:
            raise ValueError('Unknown mode')
