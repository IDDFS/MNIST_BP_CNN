import decodeMNIST
import nueralnet
import time

trainImages = decodeMNIST.decode_idx3_ubyte(decodeMNIST.train_images_idx3_ubyte_file)
trainLabels = decodeMNIST.decode_idx1_ubyte(decodeMNIST.train_labels_idx1_ubyte_file)
testImages = decodeMNIST.decode_idx3_ubyte(decodeMNIST.test_images_idx3_ubyte_file)
testLabels = decodeMNIST.decode_idx1_ubyte(decodeMNIST.test_labels_idx1_ubyte_file)
# 使用decodeMNIST文件解析MNIST数据集并将结果保存在相关数组中

input_size = 784
hidden_size1 = 50
hidden_size2 = 40
output_size = 10
batch_size = 50
learn_rate = 1
lambda_ = 0.001
epochs = 10

net = nueralnet.NueralNetwork(input_size, hidden_size1, hidden_size2, output_size, learn_rate, lambda_)
Traning_start = time.time()
net.train(trainImages, trainLabels, testImages, testLabels, batch_size, epochs)
print("Total Training Time = {0}".format(time.time() - Traning_start))
print("End")