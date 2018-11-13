---
layout: post
title: "循环神经网络RNN介绍"
tags: [深度学习]
date: 2018-11-13
---
RNN（Recurrent Neural Network）是用于处理序列数据的神经网络，它的网络结构如下图所示
<img src="/img/rnn.PNG">

$$\bar{s}_t=\Phi(\bar{x}_tW_x+\bar{s}_{t-1}W_s)$$

$$\bar{s}'_t$$
