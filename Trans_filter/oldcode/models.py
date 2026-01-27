import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.d_model = d_model  # Add d_model attribute for easy access

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)

        context = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.wo(context)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_model * 4)  # d_ff is often 4*d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

class PositionalEncoding(nn.Module):
    """
    为输入序列加入位置信息。Transformer本身不感知顺序，因此需要位置编码。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x 的形状: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
class TUNN(nn.Module):
    def __init__(self, input_length, d_model=8, num_heads=8):
        super(TUNN, self).__init__()
        self.input_length = input_length
        self.d_model = d_model

        self.initial_conv = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        encoder_channels = [d_model, d_model * 2, d_model * 4, d_model * 8, d_model * 16]

        # Following Table 4-2 architecture:
        # Encoder 0: (input_length)x1 -> (input_length)x8 -> (input_length/2)x16
        self.encoder_blocks.append(TransformerBlock(encoder_channels[0], num_heads))  # input_length, d_model=8
        self.downsamples.append(
            nn.Conv1d(encoder_channels[0], encoder_channels[1], kernel_size=2, stride=2))  # 8 -> 16 channels

        # Encoder 1: (input_length/2)x16 -> (input_length/4)x32
        self.encoder_blocks.append(TransformerBlock(encoder_channels[1], num_heads))  # input_length/2, d_model=16
        self.downsamples.append(
            nn.Conv1d(encoder_channels[1], encoder_channels[2], kernel_size=2, stride=2))  # 16 -> 32 channels

        # Encoder 2: (input_length/4)x32 -> (input_length/8)x64
        self.encoder_blocks.append(TransformerBlock(encoder_channels[2], num_heads))  # input_length/4, d_model=32
        self.downsamples.append(
            nn.Conv1d(encoder_channels[2], encoder_channels[3], kernel_size=2, stride=2))  # 32 -> 64 channels

        # Encoder 3: (input_length/8)x64 -> (input_length/16)x128
        self.encoder_blocks.append(TransformerBlock(encoder_channels[3], num_heads))  # input_length/8, d_model=64
        self.downsamples.append(
            nn.Conv1d(encoder_channels[3], encoder_channels[4], kernel_size=2, stride=2))  # 64 -> 128 channels

        # Bottleneck (input_length/16)x128
        self.bottleneck = TransformerBlock(encoder_channels[4], num_heads)  # input_length/16, d_model=128

        # Decoder blocks (Upsample + Concat Skip + TransformerBlock)
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Decoder 0 (corresponds to skip 3):
        # Input to upsample[0] is bottleneck (input_length/16)x128
        # Output of upsample[0] is (input_length/8)x64
        self.upsamples.append(
            nn.ConvTranspose1d(encoder_channels[4], encoder_channels[3], kernel_size=2, stride=2, output_padding=1))
        # Input to decoder_blocks[0] is (upsample[0] output) + (skip_connections[3]) = 64 + 64 = 128
        self.decoder_blocks.append(TransformerBlock(encoder_channels[3] + encoder_channels[3], num_heads))

        # Decoder 1 (corresponds to skip 2):
        # Input to upsample[1] is output of decoder_blocks[0] = (input_length/8)x128
        # Output of upsample[1] is (input_length/4)x96 (d_model * 12)
        self.upsamples.append(
            nn.ConvTranspose1d(encoder_channels[3] + encoder_channels[3], d_model * 12, kernel_size=2, stride=2,
                               output_padding=0))
        # Input to decoder_blocks[1] is (upsample[1] output) + (skip_connections[2]) = 96 + 32 = 128
        self.decoder_blocks.append(TransformerBlock(d_model * 12 + encoder_channels[2], num_heads))

        # Decoder 2 (corresponds to skip 1):
        # Input to upsample[2] is output of decoder_blocks[1] = (input_length/4)x128
        # Output of upsample[2] is (input_length/2)x48 (d_model * 6)
        self.upsamples.append(
            nn.ConvTranspose1d(d_model * 12 + encoder_channels[2], d_model * 6, kernel_size=2, stride=2,
                               output_padding=1))
        # Input to decoder_blocks[2] is (upsample[2] output) + (skip_connections[1]) = 48 + 16 = 64
        self.decoder_blocks.append(TransformerBlock(d_model * 6 + encoder_channels[1], num_heads))

        # Decoder 3 (corresponds to skip 0):
        # Input to upsample[3] is output of decoder_blocks[2] = (input_length/2)x64
        # Output of upsample[3] is (input_length)x32 (d_model * 4)
        self.upsamples.append(
            nn.ConvTranspose1d(d_model * 6 + encoder_channels[1], d_model * 4, kernel_size=2, stride=2,
                               output_padding=0))
        # Input to decoder_blocks[3] is (upsample[3] output) + (skip_connections[0]) = 32 + 8 = 40
        self.decoder_blocks.append(TransformerBlock(d_model * 4 + encoder_channels[0], num_heads))

        # Final Convolutional layers (Table 4-2)
        # From (input_length)x40 -> (input_length)x20 -> (input_length)x10 -> (input_length)x1
        self.final_conv1 = nn.Conv1d(d_model * 4 + encoder_channels[0], 20, kernel_size=1)
        self.final_conv2 = nn.Conv1d(20, 10, kernel_size=1)
        self.final_conv3 = nn.Conv1d(10, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch_size, 1, sequence_length) for Conv1d

        x = self.relu(self.initial_conv(x)).permute(0, 2, 1)  # (B, L, D_model)

        skip_connections = []
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)  # Apply Transformer block (B, L_curr, D_curr)
            skip_connections.append(x)  # Save for skip connection

            x = x.permute(0, 2, 1)  # Permute for Conv1D: (B, D_curr, L_curr)
            x = self.downsamples[i](x)  # Apply Downsampling (B, D_next, L_next)
            x = x.permute(0, 2, 1)  # Permute back for Transformer: (B, L_next, D_next)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(len(self.decoder_blocks)):
            # Upsample
            x = x.permute(0, 2, 1)  # (B, D_curr, L_curr)
            # No explicit output_size or output_padding calculation needed if stride=kernel_size=2 and input is even
            # ConvTranspose1d(..., output_padding=1) for odd input lengths to match
            x = self.upsamples[i](x)  # (B, D_next, L_next) (length and channel changed)
            x = x.permute(0, 2, 1)  # (B, L_next, D_next)

            skip_data = skip_connections[-(i + 1)]

            # Ensure sequence lengths match before concatenation (a common U-Net issue)
            if x.shape[1] != skip_data.shape[1]:
                min_len = min(x.shape[1], skip_data.shape[1])
                x = x[:, :min_len, :]
                skip_data = skip_data[:, :min_len, :]

            # Concatenate skip connection on the feature dimension
            x = torch.cat([x, skip_data], dim=-1)  # (B, L_matched, D_combined)

            # Apply Decoder Transformer Block
            x = self.decoder_blocks[i](x)

        # Final Conv layers to get back to (B, 1, L)
        x = x.permute(0, 2, 1)  # (B, D_final, L)
        x = self.relu(self.final_conv1(x))
        x = self.relu(self.final_conv2(x))
        denoised_spectrum = self.final_conv3(x)  # (B, 1, L)

        return denoised_spectrum


# =======================================================================
# 经过重构和优化的 TUNN 模型
# =======================================================================
class TUNN2(nn.Module):
    """
    一个基于Transformer和U-Net架构的混合模型，用于一维信号去噪。

    改进点:
    1.  **更合理的维度**: 默认d_model=64, num_heads=8, 保证了注意力头的有效性。
    2.  **位置编码**: 明确加入了PositionalEncoding来捕捉序列顺序。
    3.  **对称化解码器**: 解码器结构与编码器镜像，更简洁、鲁棒。
    4.  **修复尺寸不匹配**: 使用Padding代替Cropping，避免信息丢失。
    """

    def __init__(self, input_length, d_model=64, num_heads=8, dropout=0.1):
        super(TUNN2, self).__init__()
        self.d_model = d_model

        # 初始卷积层，将输入信号从1通道映射到d_model通道
        self.initial_conv = nn.Conv1d(1, d_model, kernel_size=11, padding=5)
        self.pos_encoder = PositionalEncoding(d_model)

        # --- 编码器 ---
        self.encoder_blocks = nn.ModuleList()
        # 定义3个下采样层
        encoder_channels = [d_model, d_model * 2, d_model * 4]
        for i in range(len(encoder_channels)):
            in_ch = encoder_channels[i - 1] if i > 0 else d_model
            out_ch = encoder_channels[i]
            # 编码器包含一个Transformer块和一个用于下采样的卷积层
            self.encoder_blocks.append(nn.ModuleList([
                TransformerBlock(in_ch, num_heads, dropout),
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            ]))

        # --- 瓶颈层 ---
        self.bottleneck = TransformerBlock(encoder_channels[-1], num_heads, dropout)

        # --- 解码器 ---
        self.decoder_blocks = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]  # [d_model*4, d_model*2, d_model]
        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1] if i < len(decoder_channels) - 1 else d_model
            # 解码器包含一个转置卷积(上采样), 一个1x1卷积(用于调整拼接后的通道), 和一个Transformer块
            self.decoder_blocks.append(nn.ModuleList([
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2),
                # 拼接后的通道数是 out_ch (来自上采样) + out_ch (来自跳跃连接)
                nn.Conv1d(out_ch * 2, out_ch, kernel_size=1),
                TransformerBlock(out_ch, num_heads, dropout)
            ]))

        # --- 输出层 ---
        self.final_conv = nn.Conv1d(d_model, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch_size, 1, sequence_length)

        # 初始卷积和位置编码
        x = self.initial_conv(x).permute(0, 2, 1)  # -> (B, L, D)
        x = self.pos_encoder(x)

        skip_connections = []
        # --- 编码过程 ---
        for transformer, downsample_conv in self.encoder_blocks:
            x = transformer(x)
            skip_connections.append(x)
            x = downsample_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # --- 瓶颈层 ---
        x = self.bottleneck(x)

        # --- 解码过程 ---
        skip_connections = skip_connections[::-1]
        for i, (upsample_conv, channel_adjust_conv, transformer) in enumerate(self.decoder_blocks):
            x = upsample_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            skip = skip_connections[i]

            # 使用Padding修复尺寸不匹配问题
            if x.shape[1] != skip.shape[1]:
                diff = skip.shape[1] - x.shape[1]
                x = F.pad(x, [0, 0, 0, diff])  # (B, L, D) -> [pad_left, pad_right] for last dim

            # 跳跃连接
            x = torch.cat([x, skip], dim=-1)  # 在特征维度上拼接 -> (B, L, D*2)

            # 1x1卷积调整通道数
            x = channel_adjust_conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # -> (B, L, D)

            x = transformer(x)

        # --- 输出 ---
        denoised_spectrum = self.final_conv(x.permute(0, 2, 1))  # -> (B, 1, L)

        return denoised_spectrum
import torch.nn.functional as F

# =======================================================================
# 模型 1: 一维卷积自编码器 (ConvAutoencoder)
# =======================================================================
class ConvAutoencoder(nn.Module):
    """
    一个经典的一维卷积自编码器，用于信号去噪。
    - 优点: 结构简单，易于理解和实现。
    - 工作原理: 通过编码器（一系列卷积和池化层）将输入信号压缩成一个低维的潜在表示，
      然后通过解码器（一系列转置卷积层）从这个潜在表示中重建出干净的信号。
      模型在学习过程中被迫捕捉信号最重要的特征，从而过滤掉噪声。
    """
    def __init__(self, input_length):
        super(ConvAutoencoder, self).__init__()
        # --- 编码器 ---
        self.encoder = nn.Sequential(
            # 输入: (B, 1, 1000)
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5), # -> (B, 16, 1000)
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2), # -> (B, 16, 500)
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3), # -> (B, 32, 500)
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2), # -> (B, 32, 250)
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2), # -> (B, 64, 250)
            nn.ReLU(True),
            nn.MaxPool1d(5, stride=5)  # -> (B, 64, 50)
        )

        # --- 解码器 ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5), # -> (B, 32, 250)
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2), # -> (B, 16, 500)
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2), # -> (B, 1, 1000)
            nn.Tanh() # 将输出限制在-1到1之间, 可以根据你的信号范围调整
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# =======================================================================
# 模型 2: 一维 U-Net (UNet1D)
# =======================================================================
class UNet1D(nn.Module):
    """
    经典U-Net模型的一维版本，非常适合信号去噪。
    - 优点: 跳跃连接（Skip Connections）允许解码器直接利用来自编码器的低层次特征，
      这有助于恢复信号的精细细节，通常比普通自编码器效果更好。
    - 工作原理: 结构与自编码器类似，但编码器的每一层输出都会通过跳跃连接直接传递给解码器
      对应层级的输入。这使得模型在重建信号时能够同时利用高层语义信息和低层细节信息。
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet1D, self).__init__()

        # --- 编码器 (下采样) ---
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool1d(2)

        # --- 瓶颈层 ---
        self.bottleneck = self.conv_block(256, 512)

        # --- 解码器 (上采样) ---
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # --- 输出层 ---
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # 瓶颈
        b = self.bottleneck(p3)

        # 解码 + 跳跃连接
        d3 = self.upconv3(b)
        # 确保尺寸匹配
        d3 = F.pad(d3, (0, e3.size(2) - d3.size(2)))
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = F.pad(d2, (0, e2.size(2) - d2.size(2)))
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = F.pad(d1, (0, e1.size(2) - d1.size(2)))
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# =======================================================================
# 模型 3: 基于RNN的序列到序列模型 (SimpleRNN)
# =======================================================================
class SimpleRNN(nn.Module):
    """
    一个基于LSTM的序列到序列模型。
    - 优点: 能够很好地捕捉信号中的时序依赖关系。
    - 工作原理: 将输入信号看作一个时间序列，使用LSTM层来读取整个序列并理解其动态变化。
      然后，一个全连接层将LSTM的输出映射回原始信号的维度，以生成去噪后的版本。
    """
    def __init__(self, input_length):
        super(SimpleRNN, self).__init__()
        self.input_length = input_length
        # 使用双向LSTM以捕捉前后文信息
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        # LSTM输出的特征维度是 hidden_size * 2 (因为是双向)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x的形状: (B, 1, 1000)
        # LSTM需要: (B, Seq_Len, Input_Size), 所以需要转换维度
        x = x.permute(0, 2, 1) # -> (B, 1000, 1)
        lstm_out, _ = self.lstm(x) # -> (B, 1000, 128)
        out = self.fc(lstm_out) # -> (B, 1000, 1)
        # 转换回原始形状: (B, 1, 1000)
        out = out.permute(0, 2, 1)
        return out


# --- New RNN + Residual + LayerNorm Model ---
class ImprovedRNN(nn.Module):
    def __init__(self, input_length):
        super(ImprovedRNN, self).__init__()
        self.input_length = input_length
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x_orig = x  # Save original input
        x = x.permute(0, 2, 1)  # (B, 1000, 1)
        lstm_out, _ = self.lstm(x)  # (B, 1000, 128)
        lstm_out = self.norm(lstm_out)
        out = self.fc(lstm_out)  # (B, 1000, 1)
        out = out.permute(0, 2, 1)  # (B, 1, 1000)
        return out + x_orig  # Residual connection


class ImprovedRNNCLAUDE(nn.Module):
    """
    改进版本1: 增强的LSTM架构
    - 添加残差连接
    - 使用LayerNorm
    - 多尺度特征提取
    """

    def __init__(self, input_length):
        super(ImprovedRNN, self).__init__()
        self.input_length = input_length

        # 多层LSTM with residual connections
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, num_layers=1,
                             batch_first=True, bidirectional=True, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1,
                             batch_first=True, bidirectional=True, dropout=0.1)

        # Layer normalization
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(128)

        # Multi-scale processing
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=15, padding=7)

        # Final layers
        self.fc1 = nn.Linear(64 * 3, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, 1, 1000)
        x = x.permute(0, 2, 1)  # -> (B, 1000, 1)

        # First LSTM layer
        lstm_out1, _ = self.lstm1(x)  # -> (B, 1000, 128)
        lstm_out1 = self.ln1(lstm_out1)

        # Second LSTM layer with residual connection
        lstm_out2, _ = self.lstm2(lstm_out1)  # -> (B, 1000, 128)
        lstm_out2 = self.ln2(lstm_out2 + lstm_out1)  # Residual connection

        # Multi-scale convolution
        conv_input = lstm_out2.permute(0, 2, 1)  # -> (B, 128, 1000)
        conv_out1 = F.relu(self.conv1(conv_input))  # -> (B, 64, 1000)
        conv_out2 = F.relu(self.conv2(conv_input))  # -> (B, 64, 1000)
        conv_out3 = F.relu(self.conv3(conv_input))  # -> (B, 64, 1000)

        # Concatenate multi-scale features
        conv_concat = torch.cat([conv_out1, conv_out2, conv_out3], dim=1)  # -> (B, 192, 1000)
        conv_concat = conv_concat.permute(0, 2, 1)  # -> (B, 1000, 192)

        # Final layers
        out = F.relu(self.fc1(conv_concat))
        out = self.dropout(out)
        out = self.fc2(out)  # -> (B, 1000, 1)

        # Convert back to original shape
        out = out.permute(0, 2, 1)  # -> (B, 1, 1000)
        return out
class SimpleRNNGRU(nn.Module):
    """
    一个基于LSTM的序列到序列模型。
    - 优点: 能够很好地捕捉信号中的时序依赖关系。
    - 工作原理: 将输入信号看作一个时间序列，使用LSTM层来读取整个序列并理解其动态变化。
      然后，一个全连接层将LSTM的输出映射回原始信号的维度，以生成去噪后的版本。
    """
    def __init__(self, input_length):
        super(SimpleRNNGRU, self).__init__()
        self.input_length = input_length
        # 使用双向LSTM以捕捉前后文信息
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        # LSTM输出的特征维度是 hidden_size * 2 (因为是双向)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x的形状: (B, 1, 1000)
        # LSTM需要: (B, Seq_Len, Input_Size), 所以需要转换维度
        x = x.permute(0, 2, 1) # -> (B, 1000, 1)
        gru_out, _ = self.gru(x) # -> (B, 1000, 128)
        out = self.fc(gru_out) # -> (B, 1000, 1)
        # 转换回原始形状: (B, 1, 1000)
        out = out.permute(0, 2, 1)
        return out

class ConvRNN(nn.Module):
    def __init__(self, input_length):
        super(ConvRNN, self).__init__()

        # 卷积前端
        self.conv_frontend = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU()
        )

        # RNN后端
        # 注意: 这里的input_size变成了卷积输出的通道数64
        self.rnn_backend = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 全连接层
        self.fc = nn.Linear(256, 1) # 128 * 2

    def forward(self, x):
        # x: (B, 1, L)
        x = self.conv_frontend(x) # -> (B, 64, L)
        x = x.permute(0, 2, 1)    # -> (B, L, 64)
        rnn_out, _ = self.rnn_backend(x)
        out = self.fc(rnn_out)
        out = out.permute(0, 2, 1)
        return out



if __name__ == '__main__':
    # Simple test of the model structure
    input_length = 1000  # Assuming your new input length
    model = TUNN(input_length)
    print(
        f"TUNN Model initialized. Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # Dummy input
    dummy_input = torch.randn(2, 1, input_length)  # Batch_size=2, Channels=1, Length=12000
    denoised_output = model(dummy_input)

    print(f"Denoised Spectrum Shape: {denoised_output.shape}")