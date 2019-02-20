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

每层的
