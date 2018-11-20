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
```python
### 读取数据
with open('sentiment/reviews.txt', 'r') as f:
       reviews = f.read() #评论
with open('sentiment/labels.txt', 'r') as f:
       labels = f.read() #情感标签
### 去掉标点符号
from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
### 获取评论和单词
reviews = all_text.split('\n')       
all_text = ' '.join(reviews)
words = all_text.split()
### 将单词转化为整数
from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True) #将单词按出现次数从多到少排序
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)} #将单词从1开始编码
reviews_ints = []
for each in reviews:
       reviews_ints.append([vocab_to_int[word] for word in each.split()])
### 将标签转化为整数
labels = labels.split('\n')
labels = np.array([1 if each == 'positive' else 0 for each in labels])
### 去掉长度为0的评论
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])
### 截取评论中的前200个单词，不足200单词的评论在左边补0
seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype=int)
for i, row in enumerate(reviews_ints):
       features[i, -len(row):] = np.array(row)[:seq_len]
```

2. 搭建网络
