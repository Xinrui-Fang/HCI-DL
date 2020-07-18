# Dataset 
* CIFAR-10
The image size of CIFAR-10 dataset is 32*32, which is too small for original ALexNet model, so I change the model of ALexNet to adapt for this dataset

# Environment && Dependency
* Windows 10
* Python 3.7
* Pytorch 1.2
* argparse

# Build and Run
The working file structure should look like this:
```
        [Layer 1]
        |---- skeleton
        |---- illustration
              |---- path 1
              |---- path 2
              |---- path 3
```


# Result
The pricison on testset can reach to **70%** after 50 epoches.
train-loss:
![train_loss](https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/AlexNet/img/train_loss.svg)
test-loss:
![train_loss](https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/AlexNet/img/test_loss.svg)
accuracy:
![train_loss](https://github.com/Xinrui-Fang/HCI-ML-with-Code/blob/master/Convolutional%20Neural%20Networks/AlexNet/img/accuracy.svg)
