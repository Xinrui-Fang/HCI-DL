# ResNet
## Dataset 
* CIFAR-10
The image size of CIFAR-10 dataset is 32*32, which is too small for original ALexNet model, so I change the model of ALexNet to adapt for this dataset

## Residual Block
按照网络模型的结构来看，模型的深度越深，准确度应该越高，最起码不会降低（假设在浅层网络之后都是恒等映射，效果是一样的）。但事实却不是这样的，当网络的深度达到一定的程度后，网络反而会退化(Degradation)， 为了解决退化问题，ResNet引入了残差模块(Residual Block)。ResNet有很多旁路支线可以将输入直接连到后面的层，使得后面的层可以直接学习残差，简化了学习难度。传统的卷积层和全连接层在信息传递时，或多或少会存在信息丢失，损耗等问题。ResNet将输入信息绕道传到输出，保护了信息的完整性。残差网络起作用的主要原因就是这些残差块学习恒等函数非常容易(F(x) = H(x) - x)。你能确定网络性能不会受到影响，很多时候甚至会提高效率。ResNet不同深度的网络结构详见下图：
<img src="https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/ResNet/img/resnet.png" width = "1000"  alt="" align=center /><br/>

## Dependency 
* Windows 10
* Python 3.7
* Pytorch 1.2
* argparse
* tensorboard

## Build and Run
1. install related dependency libraries.
1. The working file structure should look like this:
```
        [Layer 1]
        |---- main.py
        |---- ResNet.py
```
## Result
The pricison on testset can reach to **84%** after 20 epoches.<br/>
train-loss:
<img src="https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/ResNet/img/train_loss.svg" width = "500"  alt="" align=center /><br/>
accuracy:
<img src="https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/ResNet/img/accuracy.svg" width = "500"  alt="" align=center /><br/>

