import numpy as np
import time

class NueralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 1, lambda_ = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.lambda_ = lambda_
        # lambda_ 正则化项参数

        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))
        # 初始化权重weights和偏置bias
        # w1, b1 -- 输入层到隐藏层
        # w2, b2 -- 隐藏层到输出层

        # Adam优化相关参数
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1E-8
        self.t = 0

        # 动量、平方梯度项，初始均设置为0
        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)

    
    def sigmoid(self, x):
        return np.longdouble(1.0 / (1.0 + np.exp(-x)))
    # Sigmoid函数
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    # Sigmoid函数的导数
    def comopute_loss(self, y, y_hat):
        m = y.shape[0]
        cross_entropy = -np.sum(y * np.log(y_hat + 1E-8)) / m
        regularization = (self.lambda_ / (2 * m)) * (np.sum(self.w1**2) + np.sum(self.w2**2))
        return cross_entropy + regularization
    # 损失函数计算：交叉熵+L2正则化

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    # 前向传递

    def backward(self, X, y, y_hat):
        m = X.shape[0]
        error = y_hat - y # 输出层误差

        self.dw2 = (self.a1.T.dot(error) + self.lambda_ * self.w2) / m 
        self.db2 = np.sum(error, axis=0) / m
        # 输出层的delta

        delta1 =  error.dot(self.w2.T) * self.sigmoid_derivative(self.z1)
        self.dw1 = (X.T.dot(delta1) + self.lambda_ * self.w1) / m
        self.db1 = np.sum(delta1, axis=0) / m
        # 隐藏层的delta
    def update(self):
        self.t += 1

        # 更新w1
        self.m_w1 = self.beta1 * self.m_w1 + (1 - self.beta1) * self.dw1
        self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * (self.dw1 ** 2)
        m_hat1_w1 = self.m_w1 / (1 - self.beta1 ** self.t)
        v_hat1_w1 = self.v_w1 / (1 - self.beta2 ** self.t)
        self.w1 -= self.lr * m_hat1_w1 / (np.sqrt(v_hat1_w1) + self.epsilon)

        # 更新b1
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * self.db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (self.db1 ** 2)
        m_hat_b1 = self.m_b1 / (1 - self.beta1 ** self.t)
        v_hat_b1 = self.v_b1 / (1 - self.beta2 ** self.t)
        self.b1 -= self.lr * m_hat_b1 / (np.sqrt(v_hat_b1) + self.epsilon)

        # 更新w2
        self.m_w2 = self.beta1 * self.m_w2 + (1 - self.beta1) * self.dw2
        self.v_w2 = self.beta2 * self.v_w2 + (1 - self.beta2) * (self.dw2 ** 2)
        m_hat1_w2 = self.m_w2 / (1 - self.beta1 ** self.t)
        v_hat1_w2 = self.v_w2 / (1 - self.beta2 ** self.t)
        self.w2 -= self.lr * m_hat1_w2 / (np.sqrt(v_hat1_w2) + self.epsilon)

        # 更新b1
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * self.db2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (self.db2 ** 2)
        m_hat_b2 = self.m_b2 / (1 - self.beta1 ** self.t)
        v_hat_b2 = self.v_b2 / (1 - self.beta2 ** self.t)
        self.b2 -= self.lr * m_hat_b2 / (np.sqrt(v_hat_b2) + self.epsilon)
    # backward -- 反向传播误差
    # update -- 利用Adam更新参数

    def test(self, test_image, test_label):
        pred = self.predict(test_image)
        acc = np.mean(pred == np.argmax(test_label, axis = 1))
        print("\tAccuracy = {0}%".format(acc*100))
    # 测试集

    def train(self, train_image, train_label, test_image = None, test_label = None, batch_size = 50, epochs = 10):
        for epoch in range(epochs):
            epoch_start = time.time()
            indices = np.random.permutation(train_image.shape[0])
            train_image_shuffled = train_image[indices]
            train_label_shaffled = train_label[indices]
            num_batches = train_image.shape[0] // batch_size
            #打乱所有训练数据，并每batch_size一个分组分为num_batches组

            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                image_batch = train_image_shuffled[start:end]
                label_batch = train_label_shaffled[start:end]
                pred = self.forward(image_batch)
                self.backward(image_batch, label_batch, pred)
                self.update()
                # 对每个batch进行训练

            print("Epoch {0}:".format(epoch + 1))
            result = self.forward(test_image)
            loss = self.comopute_loss(test_label, result)
            print("\tLoss = {0}".format(loss))
            self.test(test_image, test_label)
            print("\tTime = {0}".format(time.time() - epoch_start))
    # 训练并测试，每次训练后输出loss和准确度

    def predict(self, image):
        y_hat = self.forward(image)
        return np.argmax(y_hat, axis=1)
    # 推理
