---
layout: post
title: "使用fastai进行图像多标签学习"
tags: [深度学习]
date: 2019-01-31
---

多标签分类(multi-label classification)项目([Data](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data))
+ 从卫星图片了解亚马逊雨林，每张图片可属于多个标签

#### F score

<div style="position:relative; left:20px;">
<table border="1" cellpadding="10">
  <tr>
    <th>标签</th>
    <th>预测为Positive(1)</th>
    <th>预测为Negative(0)</th>
  </tr>
  <tr>
    <td><strong>真值为Positive(1)</strong></td>
    <td>TP</td>
    <td>FN</td>
  </tr>
  <tr>
    <td><strong>真值为Negative(0)</strong></td>
    <td>FP</td>
    <td>TN</td>
  </tr>
</table>
<div>例如真实标签是(1,0,1,1,0,0), 预测标签是(1,1,0,1,1,0), 则TP=2, FN=1, FP=2, TN=1</div>
</div>

1. 计算Precision: $$P=\frac{TP}{TP+FP}$$   
2. 计算Recall: $$R=\frac{TP}{TP+FN}$$   
3. 计算F score: $$F=\frac{(1+\beta^2)PR}{\beta^2P+R}$$
<br>
<span style="position:relative; left:30px;">$$\beta$$越小, F score中P的权重越大; $$\beta$$等于0时F score就变为P</span>
<br>
<span style="position:relative; left:30px;">$$\beta$$越大, F score中R的权重越大; $$\beta$$趋于无穷大时F score就变为R</span>
<br>


项目代码
```python
from fastai.vision import *
path=Path('data/planet')
### Transformations for data augmentation
###     flip_vert表示上下翻转，因为是卫星图像所以打开这一项
###     max_warp是用来模拟图片拍摄时的远近和方位的不同，因为是卫星图像所以关闭这一项
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
### Use data block API instead of ImageDataBunch for data preparation
###     1. 从对应的文件夹下读取csv文件中列出的图片名称(默认为第一列)
###     2. train/validation split         3. 读取图片对应的标签(默认为第二列)
###     4. data augmentation and resize   5. 生成DataBunch并对数据进行标准化
np.random.seed(42)
src = ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')  \
                   .random_split_by_pct(0.2) \
                   .label_from_df(label_delim=' ')
data = src.transform(tfms, size=128) \
          .databunch(bs=64).normalize(imagenet_stats)
### Metrics
### 由于是多标签分类，不适合简单地使用准确率，这里采用两种评价方式
###     1. accuracy_thresh: 将分类概率大于thresh的标签设为1, 否则设为0; 同target比较计算标签的准确率
###     2. fbeta: 将分类概率大于thresh的标签设为1, 否则设为0; 计算每个样本的F score并平均
###               F score的计算方法见正文部分
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, beta=2, thresh=0.2)               
### Model
arch = models.resnet50 #使用Resnet 50进行迁移学习
learn = create_cnn(data, arch, metrics=[acc_02, f_score])
learn.lr_find()
learn.recorder.plot() #左上图
lr = 0.01 #from lr_find
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50')
### Fine-tune the whole model
learn.unfreeze()
learn.lr_find()
learn.recorder.plot() #右上图
learn.fit_one_cycle(5, slice(1e-6, lr/10))
learn.save('stage-2-rn50')
### Use bigger images to train(128*128 to 256*256)
###     Starting training on small images for a few epochs, then switching to bigger images,
###     and continuing training is an effective way to avoid overfitting
data = src.transform(tfms, size=256).databunch(bs=32).normalize(imagenet_stats)
learn.data = data
learn.freeze()
learn.lr_find()
learn.recorder.plot() #左下图
lr=1e-3
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-256-rn50')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot() #中下图
learn.fit_one_cycle(5, slice(1e-5, lr/2))
learn.recorder.plot_losses() #右下图(最近一次fit_one_cycle的train/validation loss)
learn.save('stage-2-256-rn50')
learn.export() #导出模型
```
![img](/img/planet.png)
