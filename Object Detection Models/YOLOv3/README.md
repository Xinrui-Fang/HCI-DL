# YOLOv3
**YOLO stands for You Only Look Once. It's an object detector that uses features learned by a deep convolutional neural network to detect an object.**

## Toturial
[How to implement a YOLO (v3) object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
</br>
[Pytorch 搭建自己的YOLO3目标检测平台](https://www.bilibili.com/video/BV1Hp4y1y788?p=11)

## Resources
[YOLOv3](https://github.com/bubbliiiing/yolo3-pytorch#Reference)

## The usage of batch
A big one amongst these problems is that if we want to process our images in batches (images in batches can be processed in parallel by the GPU, leading to speed boosts), we need to have all images of fixed height and width. This is needed to concatenate multiple images into a large batch (concatenating many PyTorch tensors into one)
