import torch
import torch.nn as nn



class RingdownCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(RingdownCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
        )

        # 动态计算卷积输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.squeeze()
        return x

# class CNNLSTMModel(nn.Module):
#     def __init__(self,
#                  cnn_output_channels=128, # CNN最后一层输出通道数，要和下面conv_layers定义一致
#                  lstm_hidden_size=64,     # LSTM隐藏层大小 (超参数)
#                  num_lstm_layers=1,       # LSTM层数 (超参数)
#                  dropout_rate=0.2):
#         super(CNNLSTMModel, self).__init__()
#
#         # 1. CNN 部分 (可以复用 RingdownCNN 的结构)
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=9, padding=4),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool1d(2),
#
#             nn.Conv1d(32, 64, kernel_size=7, padding=3),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool1d(2),
#
#             # 确保最后一层输出通道数与 cnn_output_channels 一致
#             nn.Conv1d(64, cnn_output_channels, kernel_size=5, padding=2),
#             nn.BatchNorm1d(cnn_output_channels),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool1d(2),
#         )
#
#         # 2. LSTM 部分
#         # input_size 对应 CNN 输出的特征维度 (通道数)
#         # batch_first=True 让输入/输出张量的形状为 (batch, seq_len, features)
#         self.lstm = nn.LSTM(input_size=cnn_output_channels,
#                             hidden_size=lstm_hidden_size,
#                             num_layers=num_lstm_layers,
#                             batch_first=True,
#                             dropout=dropout_rate if num_lstm_layers > 1 else 0, # 仅在多层LSTM时应用层间dropout
#                             bidirectional=False) # 可以尝试设置为 True 做双向LSTM
#
#         # 3. 全连接层部分
#         # 输入维度是 LSTM 的隐藏层大小 (如果是双向LSTM，则是 2 * lstm_hidden_size)
#         fc_input_size = lstm_hidden_size # 如果是双向，改为 2 * lstm_hidden_size
#         self.fc_layers = nn.Sequential(
#             nn.Linear(fc_input_size, 32), # 第一个FC层
#             nn.BatchNorm1d(32),           # 注意这里 BatchNorm1d 作用于特征维度
#             nn.LeakyReLU(0.1),
#             nn.Dropout(dropout_rate),
#             nn.Linear(32, 1)              # 输出层，预测 tau 值
#         )
#
#     def forward(self, x):
#         # 输入 x 形状: (batch_size, sequence_length)
#         x = x.unsqueeze(1)  # 添加通道维度 -> (batch_size, 1, sequence_length)
#
#         # 1. 通过 CNN 层
#         # 输出形状: (batch_size, cnn_output_channels, cnn_output_seq_len)
#         # cnn_output_seq_len 取决于卷积核、步长、填充和池化
#         cnn_out = self.conv_layers(x)
#
#         # 2. 调整形状以输入 LSTM
#         # LSTM 需要 (batch_size, seq_len, features)
#         # 当前是 (batch_size, features, seq_len)，需要换轴
#         # permute(0, 2, 1) 将第1维和第2维交换
#         lstm_input = cnn_out.permute(0, 2, 1)
#
#         # 3. 通过 LSTM 层
#         # lstm_out 形状: (batch_size, seq_len, hidden_size) (或 2*hidden_size if bidirectional)
#         # h_n 形状: (num_layers * num_directions, batch_size, hidden_size) - 最后一个时间步的隐藏状态
#         # c_n 形状: (num_layers * num_directions, batch_size, hidden_size) - 最后一个时间步的细胞状态
#         lstm_out, (h_n, c_n) = self.lstm(lstm_input)
#
#         # 4. 获取 LSTM 的最终输出用于预测
#         # 我们通常使用最后一个时间步的隐藏状态 h_n
#         # 如果 LSTM 是多层的 (num_lstm_layers > 1)，h_n 包含所有层的最后一个隐藏状态
#         # 我们需要最后一层的最后一个隐藏状态
#         # h_n[-1] 会获取最后一层（如果是双向，则是前向和后向拼接后的）隐藏状态
#         # 如果是双向LSTM: h_n 是 (num_layers*2, batch, hidden_size)，取 h_n[-2,:,:] 和 h_n[-1,:,:] 拼接
#         if self.lstm.bidirectional:
#              # 取最后前向层和最后后向层的隐藏状态拼接 (需要确认h_n的层索引和方向排布)
#              # 一个简单的方法是直接使用lstm_out的最后一个时间步输出，它包含了双向信息
#              last_lstm_output = lstm_out[:, -1, :] # 取序列最后一个时间步的输出
#              fc_input = last_lstm_output
#         else:
#             # 对于单向LSTM，取最后一层的最后一个隐藏状态
#             last_hidden_state = h_n[-1] # 形状: (batch_size, lstm_hidden_size)
#             fc_input = last_hidden_state
#
#         # 5. 通过全连接层
#         output = self.fc_layers(fc_input) # 形状: (batch_size, 1)
#
#         # 6. 去掉最后一个维度
#         output = output.squeeze() # 形状: (batch_size)
#
#         return output