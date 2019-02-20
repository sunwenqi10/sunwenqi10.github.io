---
layout: post
title: "神经网络介绍"
tags: [深度学习]
date: 2019-02-19
---

神经网络结构如下图所示（不失一般性，这里仅考虑二分类和回归问题）：

![img](/img/nn.PNG)

假设训练数据共有$$m$$个，训练数据集可由矩阵$$X=\begin{bmatrix}\begin{smallmatrix}\vdots&\vdots&\cdots&\vdots\\\vec{x}^{(1)}&\vec{x}^{(2)}&\cdots&\vec{x}^{(m)}\\\vdots&\vdots&\cdots&\vdots\end{smallmatrix}\end{bmatrix}$$表示，X为p行m列的矩阵（p为特征数）。

假设从输入层到输出层依次记为第$$0,1,2,...,L$$层，每层的节点数记为$$n_0,n_1,n_2,...,n_L$$，可以看出$$n_0=p$$，$$n_L=1$$（这里仅考虑二分类和回归问题）

第$$l$$层（$$l=1,2,...,L$$）的权重$$W^{[l]}$$为$$n_l$$行$$n_{l-1}$$列的矩阵，$$b^{[l]}$$为$$n_l$$行1列的矩阵

第$$l$$层（$$l=1,2,...,L$$）使用激活函数前的值$$Z^{[l]}$$为$$n_l$$行$$m$$列的矩阵，使用激活函数后的值$$A^{[l]}$$为$$n_l$$行$$m$$列的矩阵

### 1. Forward Propagation

线性部分：$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$$（注：$$A^{[0]}=X$$）

非线性部分：$$A^{[l]}=g(Z^{[l]})$$（$$g$$为激活函数）
+ 本文隐藏层的激活函数使用relu，可减轻梯度消失问题
+ 若为二分类问题，输出层的激活函数使用sigmoid；若为回归问题，输出层不使用激活函数，即$$A^{[L]}=Z^{[L]}$$

### 2. Loss Function

若为回归问题，损失函数可写为$$\mathcal{J}=\frac{1}{2m}\sum\limits_{i = 1}^{m}(a^{[L] (i)}-y^{(i)})^2$$，其中$$a^{[L] (i)}$$为第$$i$$个样本的预测值（即$$A^{[L]}$$的$$i$$列），$$y^{(i)}$$为第$$i$$个样本的真实值

若为二分类问题，损失函数可写为$$\mathcal{J}=-\frac{1}{m} \sum\limits_{i = 1}^{m} [y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)]$$

### 3. Backward Propagation

记$$dA^{[l]}=\frac{\partial \mathcal{J} }{\partial A^{[l]}}$$，则可推出以下公式：
+ $$dZ^{[l]}=\frac{\partial \mathcal{J} }{\partial Z^{[l]}}=dA^{[l]}* g'(Z^{[l]})$$
+ $$dW^{[l]} = \frac{\partial \mathcal{J} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T}$$
+ $$db^{[l]} = \frac{\partial \mathcal{J} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}$$
+ $$dA^{[l-1]} = \frac{\partial \mathcal{J} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]}$$
