---
layout: post
title: "GAN应用于半监督学习"
tags: [深度学习]
date: 2019-01-15
---

使用的数据集为[the Street View House Numbers(SVHN) dataset](http://ufldl.stanford.edu/housenumbers/)

为了建立一个半监督学习的情景，这里仅使用前1000个训练数据的标签，并且将GAN的判别器由二分类变为多分类，针对此数据，共分为11类（10个真实数字和虚假图像）

### 代码示例

代码的整体结构同前一篇博客[生成对抗网络GAN介绍](https://sunwenqi10.github.io/blog/2019/01/14/Introduction_To_Generative_Adversarial_Network)，这里仅注释有改动的部分

针对该网络更为细节的改进参考文章[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)以及对应的[github仓库](https://github.com/openai/improved-gan)

1. 数据处理
```python
import pickle as pkl
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
data_dir = 'data/'
trainset = loadmat(data_dir + 'svhntrain_32x32.mat')
testset = loadmat(data_dir + 'svhntest_32x32.mat')
def scale(x, feature_range=(-1, 1)):
       x = ((x - x.min())/(255 - x.min()))    
       min, max = feature_range
       x = x * (max - min) + min
       return x
class Dataset:
       def __init__(self, train, test, val_frac=0.5, shuffle=True, scale_func=None):
           split_idx = int(len(test['y'])*(1 - val_frac))
           self.test_x, self.valid_x = test['X'][:,:,:,:split_idx], test['X'][:,:,:,split_idx:]
           self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
           self.train_x, self.train_y = train['X'], train['y']
           ###################
           # For the purpose of semi-supervised learn, pretend that there are only 1000 labels
           # Use this mask to say which labels will allow to use
           self.label_mask = np.zeros_like(self.train_y)
           self.label_mask[0:1000] = 1
           ###################
           self.train_x = np.rollaxis(self.train_x, 3)
           self.valid_x = np.rollaxis(self.valid_x, 3)
           self.test_x = np.rollaxis(self.test_x, 3)
           if scale_func is None:
               self.scaler = scale
           else:
               self.scaler = scale_func
           self.train_x = self.scaler(self.train_x)
           self.valid_x = self.scaler(self.valid_x)
           self.test_x = self.scaler(self.test_x)
           self.shuffle = shuffle   
       def batches(self, batch_size, which_set="train"):
           ###################
           # Semi-supervised learn need both train data and validation(test) data   
           # Semi-supervised learn need both images and labels
           ###################
           x_name = which_set + "_x"
           y_name = which_set + "_y"
           num_examples = len(getattr(self, y_name))
           if self.shuffle:
               idx = np.arange(num_examples)
               np.random.shuffle(idx)
               setattr(self, x_name, getattr(self, x_name)[idx])
               setattr(self, y_name, getattr(self, y_name)[idx])
               if which_set == "train":
                   self.label_mask = self.label_mask[idx]
           dataset_x = getattr(self, x_name)
           dataset_y = getattr(self, y_name)
           for ii in range(0, num_examples, batch_size):
               x = dataset_x[ii:ii+batch_size]
               y = dataset_y[ii:ii+batch_size]
               if which_set == "train":
                   ###################
                   # When use the data for training, need to include the label mask
                   # Pretend don't have access to some of the labels
                   ###################
                   yield x, y, self.label_mask[ii:ii+batch_size]
               else:
                   yield x, y
dataset = Dataset(trainset, testset)
```
