import torch
import torch.nn as nn
import torch.optim as optim

# 固定随机种子，保证每次运行结果更稳定
torch.manual_seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("当前设备：", device)

# 1. 构造 XOR 数据集
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], device=device)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], device=device)

# 2. 定义神经网络
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

net = XORNet().to(device)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.5)

# 4. 训练网络
epochs = 10000

for epoch in range(epochs):
    outputs = net(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 5. 测试网络
print("\n训练完成，开始测试：")
with torch.no_grad():
    predictions = net(X)
    print("原始输出：")
    print(predictions.cpu())

    print("\n四舍五入后的结果：")
    print(torch.round(predictions).cpu())