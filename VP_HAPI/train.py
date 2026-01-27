import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. 读取数据
X = np.load('X_absorbance_samples.npy')   # shape: (num_samples, 100)
Y = np.load('Y_line_strengths.npy')       # shape: (num_samples, 1)

# 转换为 Tensor 并标准化（log(S) 更稳定）
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(np.log10(Y), dtype=torch.float32).to(device)  # 用 log10(S) 更适合作为回归目标

# 划分训练集与验证集
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. 定义模型
class MLP(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 预测 log(S)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 训练循环
num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, Y_val)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# 5. 可视化损失
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log10(S))')
plt.legend()
plt.grid(True)
plt.title('Training Curve')
plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()

# 6. 保存模型
torch.save(model.state_dict(), 'mlp_model_line_strength.pth')
print("模型已保存为：mlp_model_line_strength.pth")
