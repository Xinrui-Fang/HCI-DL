# Dataset 
* CIFAR-10
The image size of CIFAR-10 dataset is 32*32, which is too small for original ALexNet model, so I change the model of ALexNet to adapt for this dataset

# Dependency 
* Windows 10
* Python 3.7
* Pytorch 1.2
* argparse
* tensorboard

# Build and Run
1. install related dependency libraries.
1. The working file structure should look like this:
```
        [Layer 1]
        |---- main.py
        |---- AlexNet.py
```


# Result
The pricison on testset can reach to **70%** after 50 epoches.<br/>
train-loss:
<img src="https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/AlexNet/img/train_loss.svg" width = "500"  alt="" align=center /><br/>
accuracy:
<img src="https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/AlexNet/img/accuracy.svg" width = "500"  alt="" align=center /><br/>
