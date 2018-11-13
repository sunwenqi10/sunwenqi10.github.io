---
layout: post
title: "循环神经网络RNN介绍"
tags: [深度学习]
date: 2018-11-13
---
RNN（Recurrent Neural Network）是用于处理序列数据的神经网络，它的网络结构如下图所示

<img src="/img/rnn.PNG">

该网络的计算过程可表示为$$\bar{s}_t=\Phi(\bar{x}_tW_x+\bar{s}_{t-1}W_s), \bar{s}'_t=\Phi(\bar{s}_tW_y+\bar{s}'_{t-1}W_s), \bar{O}_t=\bar{s}'_tW_y$$，其中$$W_x,W_s,W_y$$为权重矩阵，$$\bar{O}_t,\bar{O}_{t+1},...$$为网络的输出

权重矩阵的计算使用BPTT算法，它的本质还是BP算法，只不过要加上基于时间的反向传播，以下图一个简单的网络为例，其中$$\bar{y}_3$$表示输出，$$\bar{d}_3$$表示实际值，$$E_3$$表示损失函数

<img src="/img/rnn1.png">

根据链式求导法则:

(1)  $$\frac{\partial{E_3}}{\partial{W_y}}=\frac{\partial{E_3}}{\partial{\bar{y}_3}}\frac{\partial{\bar{y}_3}}{\partial{W_y}}$$

(2) $$$$
