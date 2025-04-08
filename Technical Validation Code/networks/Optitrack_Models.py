import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet34_Weights
from torch.autograd import Function

'''
1DCNN
'''
class One_DCNN_CaptureData(nn.Module):
    def __init__(self):
        super(One_DCNN_CaptureData, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 320)  # 修改输入维度为 512
        self.fc2 = nn.Linear(320, 80)
        self.fc3 = nn.Linear(80, 37)

    def forward(self, x):
        x = x.transpose(1, 2)  # 将形状从 (batch_size, 298, 3) 转换为 (batch_size, 3, 298)
        x = F.relu(self.conv1(x))        # (batch_size, 64, 148)
        x = self.pool1(x)                # (batch_size, 64, 74)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))        # (batch_size, 128, 36)
        x = self.pool2(x)                # (batch_size, 128, 18)
        x = self.bn2(x)

        x = F.relu(self.conv3(x))        # (batch_size, 128, 8)
        x = self.pool3(x)                # (batch_size, 128, 4)
        x = self.bn3(x)

        feature = x.view(x.size(0), -1)  # 展平张量，形状为 (batch_size, 512)
        x = self.dropout(feature)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
Transformer 模型
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 可学习的位置编码
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.position_embedding)  # 初始化

    def forward(self, x):
        x = x + self.position_embedding[:, :x.size(1), :]
        return self.dropout(x)

class CNN_TransformerTimeSeries(nn.Module):
    def __init__(self, feature_dim=3, n_classes=37, n_heads=8, n_layers=3,
                 hidden_dim=128, dropout=0.5):
        super(CNN_TransformerTimeSeries, self).__init__()
        # 一维卷积层
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        # Transformer 编码器
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                                   dim_feedforward=hidden_dim * 4,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, feature_dim, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, seq_len, hidden_dim)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 对序列维度进行平均池化
        x = self.fc(x)
        return x

class LSTMModel_Capture(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=37, dropout_rate=0.1):
        super(LSTMModel_Capture, self).__init__()
        # 定义第一层 LSTM
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # 定义第二层 LSTM
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)
        # 定义全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过第一层 LSTM
        out, _ = self.lstm1(x)
        # 通过第二层 LSTM
        out, _ = self.lstm2(out)
        # 通过 Dropout 层
        out = self.dropout(out[:, -1, :])  # 取序列的最后一个时间步的输出
        # 通过全连接层
        out = self.fc(out)
        return out


class BiLSTMModel_Capture(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=37, dropout_rate=0.5):
        super(BiLSTMModel_Capture, self).__init__()

        # 定义第一层双向 LSTM
        self.bilstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               bidirectional=True, batch_first=True)
        # 定义第二层双向 LSTM
        self.bilstm2 = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim,
                               bidirectional=True, batch_first=True)
        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)
        # 定义全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x 的形状为 (batch_size, sequence_length, input_dim)
        # 通过第一层双向 LSTM
        out, _ = self.bilstm1(x)
        # 通过第二层双向 LSTM
        out, _ = self.bilstm2(out)
        # 取最后一个时间步的输出
        out = out[:, -1, :]  # 形状为 (batch_size, hidden_dim * 2)
        # 通过 Dropout 层
        out = self.dropout(out)
        # 通过全连接层
        out = self.fc(out)
        return out

class OneD_ConvLSTMModel_Capture(nn.Module):
    def __init__(self, input_dim=3, conv_channels=100, conv_kernel_size=10, conv_stride=1,
                 pool_size=3, pool_stride=3, hidden_dim=512, output_dim=37, dropout_rate=0.5):
        super(OneD_ConvLSTMModel_Capture, self).__init__()

        # 定义第一个 1D 卷积层
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding="same")
        self.relu1 = nn.ReLU()

        # 定义第二个 1D 卷积层
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding="same")
        self.relu2 = nn.ReLU()

        # 定义 1D 最大池化层
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

        # 定义第一层 LSTM
        self.lstm1 = nn.LSTM(input_size=conv_channels, hidden_size=hidden_dim,
                             batch_first=True)

        # 定义第二层 LSTM
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                             batch_first=True)

        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)

        # 定义全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 调整形状为 [BatchSize, Channels, Sequence]
        x = x.permute(0, 2, 1)

        # 通过卷积层和激活函数
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        # 通过池化层
        x = self.maxpool(x)  # 池化后形状为 [BatchSize, conv_channels, Reduced_Sequence]

        # 调整形状为 [BatchSize, Reduced_Sequence, conv_channels]
        x = x.permute(0, 2, 1)

        # 通过 LSTM 层
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Dropout 层
        x = self.dropout(x[:, -1, :])  # 取序列的最后一个时间步

        # 全连接层
        x = self.fc(x)

        return x

class OneDCNN_BiLSTM_Capture(nn.Module):
    def __init__(self, input_dim=3, conv_channels=100, conv_kernel_size=10, conv_stride=1,
                 pool_size=3, pool_stride=3, hidden_dim=512, output_dim=37, dropout_rate=0.5):
        super(OneDCNN_BiLSTM_Capture, self).__init__()

        # Define the first 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding="same")
        self.relu1 = nn.ReLU()

        # Define the second 1D convolutional layer
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding="same")
        self.relu2 = nn.ReLU()

        # Define the 1D max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

        # Define the first LSTM layer
        self.lstm1 = nn.LSTM(input_size=conv_channels, hidden_size=hidden_dim,
                             bidirectional=True, batch_first=True)

        # Define the second LSTM layer
        self.lstm2 = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim,
                             bidirectional=True, batch_first=True)

        # Define the Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Reshape to [BatchSize, Channels, Sequence]
        x = x.permute(0, 2, 1)

        # Pass through convolutional layers and activation functions
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        # Pass through pooling layer
        x = self.maxpool(x)  # Shape after pooling: [BatchSize, conv_channels, Reduced_Sequence]

        # Reshape to [BatchSize, Reduced_Sequence, conv_channels]
        x = x.permute(0, 2, 1)

        # Pass through LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Apply Dropout
        x = self.dropout(x[:, -1, :])  # Take the output from the last time step

        # Pass through fully connected layer
        x = self.fc(x)

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
        # Modify the first convolutional layer to adapt to 3-channel input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Modify each ResNet module to include SE module
        self.resnet.layer1 = self._make_se_layer(self.resnet.layer1)
        self.resnet.layer2 = self._make_se_layer(self.resnet.layer2)
        self.resnet.layer3 = self._make_se_layer(self.resnet.layer3)
        self.resnet.layer4 = self._make_se_layer(self.resnet.layer4)

        # Modify the fully connected layer to match the number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        # Add dropout layer
        self.dropout = nn.Dropout(0.5)

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

        # Reshape input from (batch_size, length, channels) to (batch_size, channels, 1, length)
        x = x.permute(0, 2, 1).unsqueeze(2)
        features = self.resnet.avgpool(self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(
            self.resnet.layer1(self.resnet.maxpool(F.relu(self.resnet.bn1(self.resnet.conv1(x)))))))))
        features = torch.flatten(features, 1)
        features = self.dropout(features)  # Apply dropout
        x = self.resnet.fc(features)
        return x