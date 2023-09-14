import torch
import torch.nn as nn
import torch.nn.functional as F
# from Model.Classifier import Classifier
from Model.ConvQuadraticOperation import ConvQuadraticOperation

class BasicBlock(nn.Module):
    # ResNet10/18的残差块
    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()
        self.conv1 = ConvQuadraticOperation(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.conv2 = ConvQuadraticOperation(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_channels)
        if strides != 1:  # 如果步长不为1, 则为下采样, shortcut要用conv1x1卷积块下采样
            self.conv3 = ConvQuadraticOperation(input_channels, num_channels, kernel_size=1, stride=strides)
            self.bn3 = nn.BatchNorm1d(num_channels)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.bn3(self.conv3(X))
        Y += X
        return F.relu(Y)

class QResNet(nn.Module):
    def __init__(self):
        super(QResNet, self).__init__()
        self.layer1 = nn.Sequential(ConvQuadraticOperation(1, 16, kernel_size=7, stride=1, padding=3),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=3, stride=2, padding=1))  # layer1做两次下采样
        self.layer2 = self._make_layer(16, 16, 2)
        self.layer3 = self._make_layer(16, 32, 2, downsample=True)
        self.layer4 = self._make_layer(32, 64, 2, downsample=True)
        self.layer5 = self._make_layer(64, 128, 2, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def _make_layer(self, input_channels, num_channels, num_blocks, block=BasicBlock, downsample=False):
        blk = []
        for i in range(num_blocks):
            if i == 0:
                if downsample:
                    blk.append(block(input_channels, num_channels, strides=2))
                else:
                    blk.append(block(input_channels, num_channels))
            else:
                blk.append(block(num_channels, num_channels))
        return nn.Sequential(*blk)

    def forward(self, x, epoch=0):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        y = self.flatten(self.avgpool(x5))
        return y


if __name__ == "__main__":
    pass
