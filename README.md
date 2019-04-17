# MCNN-pytorch
This is an simple and clean implemention of CVPR 2016 paper ["Single-Image Crowd Counting via Multi-Column Convolutional Neural Network."](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)  
# Installation
&emsp;1. Install pytorch  
&emsp;2. Install visdom  
```pip
pip install visdom
```
&emsp;3. Clone this repository  
```git
git clone https://github.com/CommissarMa/MCNN-pytorch.git
```
We'll call the directory that you cloned MCNN-pytorch as ROOT.
# Data Setup
&emsp;1. Download ShanghaiTech Dataset from
Dropbox: [link](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0) or Baidu Disk: [link](http://pan.baidu.com/s/1nuAYslz)  
&emsp;2. Put ShanghaiTech Dataset in ROOT and use "data_preparation/k_nearest_gaussian_kernel.py" to generate ground truth density-map. (Mind that you need modify the root_path in the main function of "data_preparation/k_nearest_gaussian_kernel.py")  
# Training
&emsp;1. Modify the root path in "train.py" according to your dataset position.  
&emsp;2. Run train.py
# Testing
&emsp;1. Modify the root path in "test.py" according to your dataset position.  
&emsp;2. Run test.py for calculate MAE of test images or just show an estimated density-map.  
# Other notes
&emsp;1. Unlike original paper, this implemention doesn't crop patches for training. We directly use original images to train mcnn model and also achieve the result as authors showed in the paper.
&emsp;2. If you are new to crowd counting, we recommand you to know [https://github.com/CommissarMa/Crowd_counting_from_scratch]{https://github.com/CommissarMa/Crowd_counting_from_scratch} first. It is an overview and tutorial of crowd counting.