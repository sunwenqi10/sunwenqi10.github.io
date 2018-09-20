---
layout: post
title: "Fast.ai深度学习第三课"
tags: [深度学习]
date: 2018-09-19
---

多标签分类(multi-label classification)项目([Data](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data))
+ 从卫星图片了解亚马逊雨林
+ 每张图片可属于多个标签
+ 损失函数可以仍然使用交叉熵，但是评判的metric使用准确率就不合适了
+ metric使用样本平均的F2 score(i.e., $$\beta=2$$时的F score)
  ```python
  ### 项目中使用的函数(伪代码)
  # th表示阈值，即当对应标签的预测值大于该值时，就认为属于该标签
  # 寻找使样本平均的F2 score最大的th，并返回对应的F2 score
  def f2(preds, targs, start=0.17, end=0.24, step=0.01):
      return max([fbeta_score(targs, (preds>th), 2, average='samples') \
                  for th in np.arange(start,end,step)])
  ```

样本平均的F score
<div style="position:relative; left:30px;">
<table border="1" cellpadding="10">
  <tr>
    <th>样本标签</th>
    <th>预测为Positive</th>
    <th>预测为Negative</th>
  </tr>
  <tr>
    <td><strong>真值为Positive</strong></td>
    <td>TP</td>
    <td>FN</td>
  </tr>
  <tr>
    <td><strong>真值为Negative</strong></td>
    <td>FP</td>
    <td>TN</td>
  </tr>
</table>
<div>例如一个真实样本的标签是(1,0,1,1,0,0), 预测的标签是(1,1,0,1,1,0), 则TP=2, FN=1, FP=2, TN=1</div>
</div>

  1. 计算每个样本的Precision: $$P=\frac{TP}{TP+FP}$$   
  2. 计算每个样本的Recall: $$R=\frac{TP}{TP+FN}$$   
  3. 计算每个样本的F score: $$F=\frac{(1+\beta^2)PR}{\beta^2P+R}$$
  <br>
  <span style="position:relative; left:30px;">$$\beta$$越小, F score中P的权重越大; $$\beta$$等于0时F score就变为P</span>
  <br>
  <span style="position:relative; left:30px;">$$\beta$$越大, F score中R的权重越大; $$\beta$$趋于无穷大时F score就变为R</span>
  <br>
  4. 对每个样本的F score进行平均

项目代码
```python
from fastai.conv_learner import *
from planet import f2 #F2 score
### 选择模型和metric
PATH = 'data/planet/'
metrics=[f2]
f_model = resnet34
label_csv = f'{PATH}train_v2.csv'
### 拆分训练和验证集
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)

#transforms_top_down：
#  RandomRotate(10), RandomLighting(0.05,0.05)
#  Rotates images by random multiples of 90 degrees, RandomFlip(horizontal)
def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms, \
                                        suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')

### 搭建模型
sz=64
#查看训练数据可使用x,y = next(iter(data.trn_dl))
#查看验证或检验数据将trn换成val或test
data = get_data(sz)
data = data.resize(int(sz*1.3), 'tmp') #将所有图片先统一转成尺寸较小的图片，加快计算速度
learn = ConvLearner.pretrained(f_model, data, metrics=metrics)
lrf=learn.lr_find() #寻找最优学习率
learn.sched.plot() #左图
### 训练模型
lr = 0.2 #由左图得出
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)

### 调整前面层次的网络参数
learn.unfreeze()
# 因为卫星图片与Imagenet上的图片差别较大，因此令前面层次的网络学习率以x3递减，而不是x10
lrs = np.array([lr/9,lr/3,lr]) #different lr for different layers
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss() #右图

### 增大图像尺寸，继续训练
sz=128
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
sz=256
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)

### 验证模型
# note: 对于多标签分类, 模型最后一层的激活函数为sigmoid而非softmax
multi_preds, y = learn.TTA() #(5, 8095, 17), (8095, 17)
preds = np.mean(multi_preds, 0) #(8095, 17)
print(f2(preds,y))
```
![img](/img/p3.png)
