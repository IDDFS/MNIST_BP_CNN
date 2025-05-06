import numpy as np
import time

class NueralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))
        # 初始化权重weights和偏置bias
        # w1, b1 -- 输入层到隐藏层
        # w2, b2 -- 隐藏层到输出层
    
    def sigmoid(self, x):
        return np.longdouble(1.0 / (1.0 + np.exp(-x)))
    # Sigmoid函数
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    # Sigmoid函数的导数

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    # 前向传递

    def backward(self, X, y, y_hat):
        error = y_hat - y # 输出层误差
        delta2 = error * self.sigmoid_derivative(self.z2) # 输出层的delta
        delta1 = np.dot(delta2, self.w2.T) * self.sigmoid_derivative(self.z1) # 隐藏层的delta

        self.w2 -= self.lr * np.dot(self.a1.T, delta2)
        self.b2 -= self.lr * np.sum(delta2, axis=0)
        self.w1 -= self.lr * np.dot(X.T, delta1)
        self.b1 -= self.lr * np.sum(delta1, axis=0)
        # 计算梯度，更新参数
    # 反向传播

    def test(self, test_image, test_label):
        pred = self.predict(test_image)
        acc = np.mean(pred == np.argmax(test_label, axis = 1))
        print("\tAccuracy = {0}%".format(acc*100))
    # 测试集

    def train(self, train_image, train_label, test_image = None, test_label = None, epochs = 30):
        for epoch in range(epochs):
            epoch_start = time.time()
            for image, label in zip(train_image, train_label):
                result = self.forward(image.reshape(1, -1))
                self.backward(image.reshape(1, -1), label.reshape(1, -1), result)
            print("Epoch {0}:".format(epoch + 1))
            result = self.forward(train_image)
            loss = np.mean(0.5 * (result - train_label)**2)
            print("\tLoss = {0}".format(loss))
            self.test(test_image, test_label)
            print("\tTime = {0}".format(time.time() - epoch_start))
    # 训练并测试，每次训练后输出loss和准确度

    def predict(self, image):
        y_hat = self.forward(image)
        return np.argmax(y_hat, axis=1)
    # 推理
