# MNIST识别（BP与CNN）

这是课程《人工智能基础》的一份作业代码。主要为使用BP与CNN对MNIST数据集进行识别。

## 文件夹介绍

`Basic_BP` -- 白板BP网络，使用最基本的Python语言实现，用到numpy库，提供对每个轮次的训练计时、测试集损失函数值、测试集准确率输出。使用Sigmoid激活函数，并采用平方损失函数。

`Improved_BP` -- 改进BP网络，在`Basic_BP`的基础上，加入SGD优化器、L2正则化，并将白板BP用的平方损失函数改成交叉熵损失函数，其他不变。

`BP_with_Adam` -- 使用Adam优化器的BP网络，是直接将`Improved_BP`里的SGD优化器换为Adam优化器，超参数选用Adam算法建议参数，其他不变。

`BP_with_2_hidden_layer` -- 有两个隐藏层的BP网络，是将`Improved_BP`直接再加一个隐藏层得到，其他不变。

`Pytorch_CNN` -- 使用Pytorch框架编写的卷积神经网络。具有两组卷积层、最大池化层以及一个带有50隐藏节点的全连接层，激活函数可选Sigmoid函数与ReLU函数。

## 代码结构

对于`Basic_BP`、`Improved_BP`、`BP_with_Adam`、`BP_with_2_hidden_layer`：

decodeMNIST.py -- 解码MNIST数据集，并可以使用PIL库将图片保存，开头需提供数据集文件夹路径；

main.py -- 训练用主程序，可在此直接更改超参数；

nueralnet.py -- 提供NueralNetwork神经网络类（使用numpy库手搓）。

对于`Pytorch_CNN`：

main.py -- 训练用主程序，使用Pytorch框架编写。

## 使用方法

对于`Basic_BP`、`Improved_BP`、`BP_with_Adam`、`BP_with_2_hidden_layer`：

在虚拟环境中装上numpy、PIL库等即可在终端中使用命令`python main.py`跑起来。

注意需自行准备数据集，并按需要修改decodeMNIST.py文件开头的数据集路径。

对于`Pytorch_CNN`：

需在虚拟环境中安装Pytorch框架，`main.py`会自行下载数据集文件，不需要手动下载。
