import decodeMNIST
import nueralnet

trainImages = decodeMNIST.decode_idx3_ubyte(decodeMNIST.train_images_idx3_ubyte_file)
trainLabels = decodeMNIST.decode_idx1_ubyte(decodeMNIST.train_labels_idx1_ubyte_file)
testImages = decodeMNIST.decode_idx3_ubyte(decodeMNIST.test_images_idx3_ubyte_file)
testLabels = decodeMNIST.decode_idx1_ubyte(decodeMNIST.test_labels_idx1_ubyte_file)
# 使用decodeMNIST文件解析MNIST数据集并将结果保存在相关数组中

input_size = 784
hidden_size = 100
output_size = 10
learn_rate = 1
epochs = 10

net = nueralnet.NueralNetwork(input_size, hidden_size, output_size, learn_rate)
net.train(trainImages, trainLabels, testImages, testLabels, epochs)
print("End")