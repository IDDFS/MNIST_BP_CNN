import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

batch_size = 50
learning_rate = 0.01
momentum = 0.5

# 数据加载与预处理，使用torch库自带的方法
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 训练集打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 测试集原样

class NueralNet(torch.nn.Module):
    def __init__(self):
        super(NueralNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            # torch.nn.ReLU(),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # input: (batch, 1, 28, 28)
        # conv: (batch, 10, 24, 24)
        # pool: (batch, 10, 12, 12)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            # torch.nn.ReLU(),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # input: (batch, 10, 12, 12)
        # conv: (batch, 20, 8, 8)
        # pool: (batch, 20, 4, 4)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10)
        )
        # 全连接层，320节点经训练后过一个50节点隐藏层再输出到10类
    def forward(self, x):
        num = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(num, -1) # 展平，下接全连接层
        x = self.fc(x)
        return x

model = NueralNet()
criterion = torch.nn.CrossEntropyLoss()
# 损失函数：交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# 使用SGD优化，带有动量优化

def train(epochs):
    for epoch in range(epochs):
        epoch_start = time.time()
        for train_images, train_labels in train_loader:
            optimizer.zero_grad()
            # 取出当前batch并训练
            pred = model(train_images)
            loss = criterion(pred, train_labels)
            loss.backward()
            optimizer.step()
        # 测试当前的准确率与loss
        print("Epoch {0}:".format(epoch + 1))
        test()
        print("\tTime = {0}".format(time.time() - epoch_start))

def test():
    correct = 0
    total = 0
    test_loss_sum = 0.0
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            outputs = model(test_images)
            loss = criterion(outputs, test_labels)
            test_loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    acc = correct / total
    test_loss = test_loss_sum / len(test_loader)
    print("\tLoss = {0}".format(test_loss))
    print("\tAccuracy = {0}%".format(acc * 100))

if __name__ == '__main__':
    train_start = time.time()
    train(epochs=10)
    print("Total Training Time = {0}".format(time.time() - train_start))
    print("End")