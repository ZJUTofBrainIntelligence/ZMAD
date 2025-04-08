import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet34_Weights
from torch.autograd import Function
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv,GINConv

from torch_scatter import scatter_add
import torch_sparse
'''
1DCNN BatchNorm版本
'''
class One_DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, stride=2)  # (13320, 64, 200)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # (13320, 64, 100)
        self.bn1 = nn.BatchNorm1d(64)  # (13320, 64, 100)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # (13320, 128, 49)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # (13320, 128, 24)
        self.bn2 = nn.BatchNorm1d(128)  # (13320, 128, 24)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)  # (13320, 128, 11)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # (13320, 128, 5)
        self.bn3 = nn.BatchNorm1d(128)  # (13320, 128, 5)
                                                                                            # x.view(x.size(0), -1)  (13320， 640)
        self.dropout = nn.Dropout(0.5)  # 正则化
        self.fc1 = nn.Linear(640, 320)  # (13320， 37)
        self.fc2 = nn.Linear(320, 80)
        self.fc3 = nn.Linear(80, 37)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)
        feature = x.view(x.size(0), -1)  # Flatten the tensor                                      #（13320， 640）
        x = self.dropout(feature)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

'''
1DCNN LayerNorm版本 
'''
# class One_DCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 定义卷积层和池化层
#         self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, stride=2)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.ln1 = nn.LayerNorm(64)  # 对通道维度进行归一化
#
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.ln2 = nn.LayerNorm(128)
#
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.ln3 = nn.LayerNorm(128)
#
#         self.dropout = nn.Dropout(0.5)
#
#         # 计算全连接层的输入尺寸
#         self.flatten_dim = 128 * 5  # 128 是通道数，5 是最终的序列长度
#         self.fc1 = nn.Linear(self.flatten_dim, 320)
#         self.fc2 = nn.Linear(320, 80)
#         self.fc3 = nn.Linear(80, 37)
#
#     def forward(self, x):
#         # 输入形状：(batch_size, sequence_length, channels)
#         x = x.transpose(1, 2)  # 转换为 (batch_size, channels, sequence_length)
#
#         # 第一层卷积块
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = x.transpose(1, 2)  # 转换为 (batch_size, sequence_length, channels)
#         x = self.ln1(x)
#         x = x.transpose(1, 2)  # 转换回 (batch_size, channels, sequence_length)
#
#         # 第二层卷积块
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = x.transpose(1, 2)
#         x = self.ln2(x)
#         x = x.transpose(1, 2)
#
#         # 第三层卷积块
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
#         x = x.transpose(1, 2)
#         x = self.ln3(x)
#         x = x.transpose(1, 2)
#
#         # 展平并通过全连接层
#         feature = x.reshape(x.size(0), -1)  # 使用 reshape 代替 view
#         x = self.dropout(feature)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
'''
1DCNN GroupNorm版本 (best）
'''
# class One_DCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 定义卷积层和池化层
#         self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, stride=2)    # 输出长度：199
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)                                 # 输出长度：99
#         self.gn1 = nn.GroupNorm(num_groups=1, num_channels=64)  # 对通道维度进行归一化
#
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # 输出长度：49
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)                                 # 输出长度：24
#         self.gn2 = nn.GroupNorm(num_groups=1, num_channels=128)
#
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2) # 输出长度：11
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)                                 # 输出长度：5
#         self.gn3 = nn.GroupNorm(num_groups=1, num_channels=128)
#
#         self.dropout = nn.Dropout(0.5)  # 正则化
#
#         # 计算全连接层的输入尺寸
#         self.flatten_dim = 128 * 5  # 128 是通道数，5 是最终的序列长度
#         self.fc1 = nn.Linear(self.flatten_dim, 320)
#         self.fc2 = nn.Linear(320, 80)
#         self.fc3 = nn.Linear(80, 37)
#
#     def forward(self, x):
#         # 输入形状：(batch_size, sequence_length, channels)
#         x = x.transpose(1, 2)  # 转换为 (batch_size, channels, sequence_length)
#
#         # 第一层卷积块
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = self.gn1(x)  # 对通道维度进行归一化
#
#         # 第二层卷积块
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = self.gn2(x)
#
#         # 第三层卷积块
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
#         x = self.gn3(x)
#
#         # 展平并通过全连接层
#         feature = x.view(x.size(0), -1)  # 展平张量，形状为 (batch_size, 128 * 5)
#         x = self.dropout(feature)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



'''
Transformer
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_dim, n_classes, n_heads=8, n_layers=3, hidden_dim=128, dropout=0.5):
        super(TransformerTimeSeries, self).__init__()
        self.input_layer = nn.Linear(feature_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, feature_dim)
        x = self.input_layer(x)
        x = x.transpose(0, 1)  # Shape: (sequence_length, batch_size, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        return self.fc(x)


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
    def __init__(self, feature_dim=8, n_classes=37, n_heads=8, n_layers=3,
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

'''
InceptionTime Network
'''

class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(InceptionModule, self).__init__()
        # 分支1：1x1卷积
        self.branch1_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding=0)
        self.branch1_gn = nn.GroupNorm(num_groups=1, num_channels=n_filters)
        # 分支2：3x3卷积
        self.branch3_conv = nn.Conv1d(in_channels, n_filters, kernel_size=3, padding=1)
        self.branch3_gn = nn.GroupNorm(num_groups=1, num_channels=n_filters)
        # 分支3：5x5卷积
        self.branch5_conv = nn.Conv1d(in_channels, n_filters, kernel_size=5, padding=2)
        self.branch5_gn = nn.GroupNorm(num_groups=1, num_channels=n_filters)
        # 分支4：最大池化 + 1x1卷积
        self.branch_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding=0)
        self.branch_pool_gn = nn.GroupNorm(num_groups=1, num_channels=n_filters)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 分支1
        branch1 = self.branch1_conv(x)
        branch1 = self.branch1_gn(branch1)
        branch1 = self.relu(branch1)

        # 分支2
        branch3 = self.branch3_conv(x)
        branch3 = self.branch3_gn(branch3)
        branch3 = self.relu(branch3)

        # 分支3
        branch5 = self.branch5_conv(x)
        branch5 = self.branch5_gn(branch5)
        branch5 = self.relu(branch5)

        # 分支4
        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        branch_pool = self.branch_pool_gn(branch_pool)
        branch_pool = self.relu(branch_pool)

        outputs = [branch1, branch3, branch5, branch_pool]
        return torch.cat(outputs, dim=1)  # 在通道维度上连接

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(InceptionBlock, self).__init__()
        self.inception1 = InceptionModule(in_channels, n_filters)
        self.inception2 = InceptionModule(4 * n_filters, n_filters)
        self.inception3 = InceptionModule(4 * n_filters, n_filters)
        out_channels = 4 * n_filters

        # 残差连接，需要匹配输入和输出通道数
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.residual_gn = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        else:
            self.residual_conv = None
            self.residual_gn = None

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
            residual = self.residual_gn(residual)
        else:
            residual = x

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x += residual  # 多级残差连接
        x = self.activation(x)
        return x

class InceptionTime(nn.Module):
    def __init__(self, n_classes=37, input_channels=8, n_filters=32, n_blocks=9):
        super(InceptionTime, self).__init__()
        self.inception_blocks = nn.ModuleList()
        for i in range(n_blocks):
            if i == 0:
                in_channels = input_channels
            else:
                in_channels = 4 * n_filters
            block = InceptionBlock(in_channels, n_filters)
            self.inception_blocks.append(block)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4 * n_filters, n_classes)

    def forward(self, x):
        # 输入形状：(batch_size, sequence_length, channels)
        x = x.transpose(1, 2)  # 转换为：(batch_size, channels, sequence_length)
        for block in self.inception_blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)  # 全局平均池化并去除最后一维
        return self.fc(x)


'''
GatedTabTransformer
'''


# Spatial Gating Unit
class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.normal_(self.spatial_proj.weight, std=1e-6)

    def forward(self, x):
        # Split the input into two parts along the last dimension
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        return u * v  # Element-wise multiplication of the two parts

# gMLP Block
class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)  # Double the projection
        self.channel_proj2 = nn.Linear(d_ffn, d_model)  # Project back to original dimension
        self.activation = nn.GELU()
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        shortcut = x  # Save input for the skip connection
        x = self.norm(x)
        x = self.channel_proj1(x)  # Project to higher dimension
        x = self.activation(x)
        x = self.sgu(x)  # Apply Spatial Gating Unit
        x = self.channel_proj2(x)  # Project back to original dimension
        return x + shortcut  # Add skip connection

# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [BatchSize, SequenceLength, Channels]
        x = x.transpose(0, 1)  # Transpose to [SequenceLength, BatchSize, Channels]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x.transpose(0, 1)  # Transpose back to [BatchSize, SequenceLength, Channels]


class GatedTabTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, seq_len, channels):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.gmlp_block = gMLPBlock(d_model, d_ff, seq_len)
        self.final_layer = nn.Linear(d_model, 37)  # Output layer for binary classification

    def forward(self, x):
        # x shape: [BatchSize, SequenceLength, Channels] -> [BatchSize, 400, 8]
        for block in self.transformer_blocks:
            x = block(x)

        x = self.gmlp_block(x)
        # Final output: use mean pooling to reduce sequence dimension for final classification/regression
        return torch.sigmoid(self.final_layer(x.mean(dim=1)))  # Reduce seq_len dimension









'''
LSTM

'''
class LSTMModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=512, output_dim=37, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
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
        out = self.dropout(out[:, -1, :])  # 取序列的最后一个输出
        # 通过全连接层
        out = self.fc(out)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(BiLSTMModel, self).__init__()

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

        # 通过第一层双向 LSTM
        out, _ = self.bilstm1(x)
        # 通过第二层双向 LSTM
        out, _ = self.bilstm2(out)
        # 通过 Dropout 层
        out = self.dropout(out[:, -1, :])  # 使用序列的最后一个时间步
        # 通过全连接层
        out = self.fc(out)
        return out




'''
1DCNN_LSTM
'''

class OneD_ConvLSTMModel(nn.Module):
    def __init__(self, input_dim = 8, conv_channels = 100, conv_kernel_size = 10, conv_stride = 1,
                 pool_size = 3, pool_stride = 3, hidden_dim = 512, output_dim = 37, dropout_rate = 0.5):
        super(OneD_ConvLSTMModel, self).__init__()

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



class OneDCNN_BiLSTM(nn.Module):
    def __init__(self, input_dim=8, conv_channels=100, conv_kernel_size=10, conv_stride=1,
                 pool_size=3, pool_stride=3, hidden_dim=512, output_dim=37, dropout_rate=0.5):
        super(OneDCNN_BiLSTM, self).__init__()

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

'''
2DCNN

'''
class TwoDCNN(nn.Module):
    def __init__(self, num_classes=37, dropout_rate=0.5):
        super(TwoDCNN, self).__init__()

        # 输入形状为 (BatchSize, 1, 400, 8)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)

        # 全连接层的输入尺寸需要根据实际数据经过卷积和池化后的输出尺寸来确定
        self.fc = nn.Linear(128 * 50 * 1, num_classes)  # 假设输出尺寸为 (50, 2)

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), 1, 400, 8)
        # print(x.shape)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

'''
DANN
'''

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        # Feature extractor
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)

        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(640, 320)
        self.fc2 = nn.Linear(320, 80)
        self.fc3 = nn.Linear(80, 37)

        # Domain classifier
        self.domain_fc1 = nn.Linear(640, 100)
        self.domain_fc2 = nn.Linear(100, 2)  # Assuming binary domain classification

    def forward(self, x, alpha=0.0):
        # Feature extraction
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)

        feature = x.view(x.size(0), -1)  # Flatten the tensor

        # Label prediction
        x = self.dropout(feature)
        class_output = self.fc1(x)
        class_output = self.fc2(class_output)
        class_output = self.fc3(class_output)

        # Domain prediction
        reverse_feature = grad_reverse(feature, alpha)
        domain_output = F.relu(self.domain_fc1(reverse_feature))
        domain_output = self.domain_fc2(domain_output)

        return class_output, domain_output






class MambaClassifier(nn.Module):
    def __init__(self, num_classes=37, input_channels=8, seq_length=400, d_model=64, n_layers=2, d_state=16, expand=2):
        super(MambaClassifier, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length

        # CNN 部分
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)

        # 残差块列表
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_state=d_state, expand=expand) for _ in range(n_layers)
        ])

        # 全连接层
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 输入形状: (B, L, C)
        x = x.transpose(1, 2)  # 转换为 (B, C, L)

        # CNN 部分
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.transpose(1, 2)  # 转换回 (B, L, d_model)

        # Mamba 残差块
        for layer in self.layers:
            x = layer(x)

        # 池化和全连接层
        x = x.mean(dim=1)  # 全局平均池化，形状变为 (B, d_model)
        logits = self.fc(x)  # 输出形状: (B, num_classes)

        return logits


class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_block = MambaBlock(d_model, d_state=d_state, expand=expand)

    def forward(self, x):
        # 残差连接
        return x + self.mamba_block(self.norm(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 深度可分离卷积
        self.depthwise_conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner  # 实现深度卷积
        )

        # 状态空间模型参数
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2)
        # 初始化 A，使用较小的值
        self.A = nn.Parameter(torch.ones(self.d_inner, self.d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 添加状态的规范化层
        self.state_norm = nn.LayerNorm([self.d_inner, self.d_state])

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x 形状: (B, L, d_model)
        B, L, _ = x.shape

        # 输入投影并拆分
        x_proj = self.in_proj(x)  # 形状: (B, L, 2 * d_inner)
        x_conv, x_residual = x_proj.chunk(2, dim=-1)

        # 卷积部分
        x_conv = x_conv.transpose(1, 2)  # 转换为 (B, d_inner, L)
        x_conv = F.gelu(self.depthwise_conv(x_conv))
        x_conv = x_conv.transpose(1, 2)  # 转换回 (B, L, d_inner)

        # 状态空间模型
        y = self.ssm(x_conv)

        # 与残差部分相乘
        y = y * F.gelu(x_residual)

        # 输出投影
        y = self.out_proj(y)

        return y


    def ssm(self, x):
        # 简化的状态空间模型实现
        B, L, d_inner = x.shape
        n = self.d_state

        # 投影并拆分 B 和 C
        x_proj = self.x_proj(x)  # 形状: (B, L, 2 * n)
        B_param, C_param = x_proj.chunk(2, dim=-1)  # 形状: (B, L, n)

        # 对 B_param 和 C_param 应用 tanh 激活函数
        B_param = torch.tanh(B_param)
        C_param = torch.tanh(C_param)

        # 初始化状态
        state = torch.zeros(B, d_inner, n, device=x.device)
        outputs = []

        # 逐步计算
        for t in range(L):
            state = state + B_param[:, t].unsqueeze(1)
            output = (state * C_param[:, t].unsqueeze(1)).sum(dim=-1) + x[:, t] * self.D

            # 检查是否有 NaN 或 Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"NaN or Inf detected in output at time step {t}")
                break

            outputs.append(output)

            # 更新状态，使用 tanh 激活函数
            state = state * torch.tanh(self.A).unsqueeze(0)

            # 对状态进行规范化
            state = self.state_norm(state)

        # 堆叠输出
        y = torch.stack(outputs, dim=1)  # 形状: (B, L, d_inner)
        return y


'''

GCN+GAT+GIN图神经网络

'''
class EMG_GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(EMG_GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GATConv(128, 256)
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        ))

        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.res_conn1 = nn.Linear(num_node_features, 128)
        self.res_conn2 = nn.Linear(128, 256)
        self.res_conn3 = nn.Linear(256, 512)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 第一层GCN
        residual = self.res_conn1(x)
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = self.dropout(x1)
        x = x1 + residual  # 残差连接

        # 第二层GAT
        residual = self.res_conn2(x)
        x2 = self.conv2(x, edge_index)
        x2 = F.leaky_relu(x2)
        x2 = self.dropout(x2)
        x = x2 + residual  # 残差连接

        # 第三层GIN
        residual = self.res_conn3(x)
        x3 = self.conv3(x, edge_index)
        x3 = F.relu(x3)  # GIN一般使用ReLU激活
        x3 = self.dropout(x3)
        x = x3 + residual  # 残差连接

        # 全局平均池化
        x = global_mean_pool(x, batch)
        # 全连接层
        x = self.fc(x)
        return x





