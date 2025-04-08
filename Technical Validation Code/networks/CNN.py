import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from networks.modules.KAN import KAN, KANLinear
# from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet34_Weights

class EMGNet(nn.Module):
    def __init__(self): 
        super(EMGNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, stride=2)       #(13320, 64, 200)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)                                    #(13320, 64, 100)
        self.bn1 = nn.BatchNorm1d(64)                                                         #(13320, 64, 100)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)     #(13320, 128, 49)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)                                    #(13320, 128, 24)
        self.bn2 = nn.BatchNorm1d(128)                                                        #(13320, 128, 24)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)    #(13320, 128, 11)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)                                    #(13320, 128, 5)
        self.bn3 = nn.BatchNorm1d(128)                                                        #(13320, 128, 5)
                                                                                              #x.view(x.size(0), -1)  (13320， 640)
        self.dropout = nn.Dropout(0.5)                                                        #正则化
        self.fc1 = nn.Linear(640, 37)                                    # (13320， 37)
        # self.fc2 = nn.Linear(256, 26)

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        # x = self.bn3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor                                      #（13320， 640）
        x = self.dropout(x)
        x = self.fc1(x)
        # x = self.fc2(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        out = self.avg_pool(x)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return residual * out

class EMGResNet(nn.Module):
    def __init__(self, num_classes=37):
        super(EMGResNet, self).__init__()
        self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # self.resnet = models.resnet34(weights=None)
        # 修改第一层卷积层，适应8个通道的输入
        self.resnet.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改每个ResNet模块以包含SE模块
        self.resnet.layer1 = self._make_se_layer(self.resnet.layer1)
        self.resnet.layer2 = self._make_se_layer(self.resnet.layer2)
        self.resnet.layer3 = self._make_se_layer(self.resnet.layer3)
        self.resnet.layer4 = self._make_se_layer(self.resnet.layer4)

        # 修改全连接层以匹配类别数量
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        # 添加 dropout 层
        self.dropout = nn.Dropout(0.3)

    def _make_se_layer(self, layer):
        se_layers = []
        for block in layer:
            se_layers.append(block)
            se_layers.append(SEModule(block.conv2.out_channels))
        return nn.Sequential(*se_layers)

    def forward(self, x, updated_params=None):
        if updated_params is not None:
            # Use the updated parameters to perform the forward pass
            idx = 0
            for name, param in self.resnet.named_parameters():
                if 'weight' in name or 'bias' in name:
                    param.data = updated_params[idx].data
                    idx += 1

        # 将输入从 (batch_size, length, channels) 转换为 (batch_size, channels, 1, length)
        x = x.permute(0, 2, 1).unsqueeze(2)
        features = self.resnet.avgpool(self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(
            self.resnet.layer1(self.resnet.maxpool(F.relu(self.resnet.bn1(self.resnet.conv1(x)))))))))
        features = torch.flatten(features, 1)
        features = self.dropout(features)  # 应用 dropout
        x = self.resnet.fc(features)
        return x


def compute_mmd(x, y, kernel='rbf'):
    def rbf_kernel(x1, x2, gamma=None):
        n1, d1 = x1.shape
        n2, d2 = x2.shape
        assert d1 == d2
        if gamma is None:
            gamma = 1.0 / d1
        x1 = x1.unsqueeze(1).expand(n1, n2, d1)
        x2 = x2.unsqueeze(0).expand(n1, n2, d2)
        return torch.exp(-gamma * ((x1 - x2) ** 2).sum(2))

    if kernel == 'linear':
        K_xx = x @ x.t()
        K_yy = y @ y.t()
        K_xy = x @ y.t()
    elif kernel == 'rbf':
        K_xx = rbf_kernel(x, x)
        K_yy = rbf_kernel(y, y)
        K_xy = rbf_kernel(x, y)
    else:
        raise ValueError("Unknown kernel type: {}".format(kernel))

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 使用 BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 使用 BatchNorm
        self.downsample = downsample
        self.stride = stride
        self.se_module = SEModule(planes, reduction)

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)  # 使用 BatchNorm
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  # 使用 BatchNorm

        if self.downsample is not None:
            identity = self.downsample(x)  # 匹配维度

        out += identity  # 残差连接
        out = self.se_module(out)  # 应用 SE 模块（在残差连接之后）
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=37, input_channels=8):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 使用 BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建残差层。
        """
        downsample = None
        # 如果输入和输出维度不一致，或者步幅不为 1，则需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),  # 使用 BatchNorm
            )

        layers = []
        # 第一个残差块，可能需要下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 其余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, updated_params=None):
        if updated_params is not None:
            # 如果提供了更新的参数，则使用它们进行前向传播
            idx = 0
            for name, param in self.named_parameters():
                if 'weight' in name or 'bias' in name:
                    param.data = updated_params[idx].data
                    idx += 1

        # 调整输入维度：(batch_size, length, channels) -> (batch_size, channels, 1, length)
        x = x.permute(0, 2, 1).unsqueeze(2)

        # 前向传播
        x = self.conv1(x)
        x = self.bn1(x)  # 使用 BatchNorm
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 第一层
        x = self.layer2(x)  # 第二层
        x = self.layer3(x)  # 第三层
        x = self.layer4(x)  # 第四层

        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc(x)  # 全连接层

        return x

def emg_resnet34(num_classes=37, input_channels=8):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                   input_channels=input_channels)
    return model


if __name__ == '__main__':
    x1 = torch.rand(50, 140, 8)
    mo = EMGNet()
    print(mo(x1).shape)