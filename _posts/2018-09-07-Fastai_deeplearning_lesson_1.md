---
layout: post
title: "Fast.ai深度学习第一课"
tags: [深度学习]
date: 2018-09-07
---

**介绍了使用fastai库将训练好的CNN模型进行猫狗分类的最基本步骤，通过以下几行代码即可完成**：

```python
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from fastai.metrics import *

arch=resnet34
sz=224
lr=0.01
epochs=2
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(lr, epochs)
```

+ PATH下必须有train和valid文件夹，每个文件夹下面必须按照分类有相应的子文件夹
+ sz是将图片统一转换成的大小
+ data是ImageClassifierData数据类型
  + data.val_y: 每个validation数据的标签
  + data.val_ds.fnames: 每个validation数据的路径(例如valid/dogs/dog.9629.jpg)   


**介绍了一种选择学习率的方法**([Cyclical Learning Rates for Training Neural Networks](http://arxiv.org/abs/1506.01186))：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在前面几次的迭代中将学习率从一个很小的值逐渐增加，选择距validation损失停止下降的点最近的有较大下降速率的点做为模型的学习率（例如下图中学习率可以选择0.01）。
```python
learn = ConvLearner.pretrained(arch, data, precompute=True)
lrf=learn.lr_find()
learn.sched.plot_lr() #左图
learn.sched.plot() #右图
```
![img](/img/p1.png)

**介绍了数据扩充(Data Augmentation)**：

```python
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
```
+ transforms_side_on: RandomRotate, RandomLighting, RandomFlip(horizontal)

```python
data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
x,y = next(iter(data.aug_dl))
x_de = data.trn_ds.denorm(x)
```
+ bs: batch_size
+ x: augmented training data(shape: bs x 3 x sz x sz); y: labels(shape: bs)
+ x_de:
  + denormalized training data(shape: bs x sz x sz x 3)
  + for each channel, x*channel_std+channel_mean
  + can be directly plotted using plt.imshow()

**介绍了如何在训练中使用数据扩充**：


1. **搭建并使用precompute=True训练模型(disable data augmentation)**
```python
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(1e-2, 1)
```
  + precompute: 是否在训练之前将所有训练数据的激活特征(即最后一个全连接层之前的特征)预先计算出来
  + precompute=True：预先计算，不做数据扩充    
  + precompute=False: 不预先计算，可做数据扩充
  + 学习速率可由函数lr_find给出

2. **使用扩充数据训练模型**
```python
learn.precompute=False
learn.fit(1e-2, 3, cycle_len=1)
learn.sched.plot_lr() #左图
learn.save('224_lastlayer') #存储模型
learn.load('224_lastlayer') #调取模型
```
  + use stochastic gradient descent with restarts(SGDR), gradually decreases the learning rate within a cycle as training progresses
  + cycle_len: number of epochs within a cycle
  + 此时fit函数中的第二个参数表示cycle的个数，而不是epochs的个数
  + SGDR的优势体现在右图（[reference](https://blog.csdn.net/suredied/article/details/80822678)）
  ![img](/img/p2.png)

3. **对模型参数进行微调**
```python
learn.unfreeze() #解冻其余层的模型参数
lr=np.array([1e-4,1e-3,1e-2]) #different lr for different layers
#################
lrf = learn.lr_find(lr/1000) #使用lr_find寻找合适的学习率
learn.sched.plot() #画出最后一层的学习率-损失曲线
#################
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.save('224_all')
learn.load('224_all')
```
  + 学习速率每次重置时，cycle内的epoch数目等于上一个cycle内的数目乘以cycle_mult

4. **使用test time augmentation(TTA)**
```python
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y) #validation accuracy
### plot confusion matrix
from sklearn.metrics import confusion_matrix
preds = np.argmax(probs, axis=1)
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)
```
  + 使用validation数据集中的原始图片以及4个augment后的数据集
  + y: target labels(不是predict labels)
