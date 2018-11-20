---
layout: post
title: "使用Tensorflow实现词嵌入和Word2Vec"
tags: [深度学习]
date: 2018-11-20
---

当处理文本中的单词时，传统的one-hot encode会产生仅有一个元素为1，其余均为0的长向量，这对于网络计算是极大地浪费，使用词嵌入技术（Embeddings）可以有效地解决这一问题。Embeddings将单词转化为整数，并且将weight matrix看作一个lookup table(如左图所示)，从而避免了稀疏向量与矩阵的直接相乘。

本文首先介绍在RNN中使用Embeddings进行情感分析（sentiment analysis），所使用的网络结构如右图所示；接着介绍一种特殊的词嵌入模型Word2Vec，用来将单词转化成包含语义解释（semantic meaning）的向量。

<img src="/img/embed.png">

### sentiment analysis

使用的数据[下载](https://pan.baidu.com/s/16tNfl50mGiga654M8tGKWQ)，其中reviews.txt中已将大写字母全部转化为小写字母

1. 数据前处理
