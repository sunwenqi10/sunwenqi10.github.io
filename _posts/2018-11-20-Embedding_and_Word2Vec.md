---
layout: post
title: "使用Tensorflow实现词嵌入和Word2Vec"
tags: [深度学习]
date: 2018-11-20
---

当处理文本中的单词时，传统的one-hot encode会产生仅有一个元素为1，其余均为0的长向量，这对于网络计算是极大地浪费，使用词嵌入技术（Embeddings）可以有效地解决这一问题。Embeddings将单词转化为整数，并且将weight matrix看作一个lookup table(如左图所示)，从而避免了稀疏向量与矩阵的直接相乘。

本文首先介绍在RNN中使用Embeddings进行情感分析（sentiment analysis），所使用的网络结构如右图所示；接着介绍一种特殊的词嵌入模型Word2Vec，用来将单词转化成包含语义解释（semantic meaning）的向量。

<img src="/img/embed.png">

### Sentiment Analysis

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
### 拆分训练、验证和测试集
split_frac = 0.8
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]
test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]
```

2. 搭建网络
```python
### 超参数设置
lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001
embed_size = 300 #Size of the embedding vectors(number of units in the embedding layer)
### Input
graph = tf.Graph() #Create the graph object
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
### Embed Layer
n_words = len(vocab_to_int) + 1 #Adding 1 because we use 0's for padding, dictionary started at 1
with graph.as_default():
       embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
       embed = tf.nn.embedding_lookup(embedding, inputs_) #3D tensor (batch_size, seq_len, embed_size)
### LSTM Layer
with graph.as_default():
       lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)    
       drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
       cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
       initial_state = cell.zero_state(batch_size, tf.float32) #Getting an initial state
### Output layer
with graph.as_default():
       outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
       predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid) #从最后一步的输出(batch_size, lstm_size)建立全连接层
### Loss function and Optimizer
with graph.as_default():
       cost = tf.losses.mean_squared_error(labels_, predictions)  
       optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
### Validation and Test Accuracy
with graph.as_default():
       correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
       accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

3. 训练网络
```python
def get_batches(x, y, batch_size=100):    
       n_batches = len(x)//batch_size
       x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
       for ii in range(0, len(x), batch_size):
           yield x[ii:ii+batch_size], y[ii:ii+batch_size]
### Train and Validation
epochs = 10
with graph.as_default():
       saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
       sess.run(tf.global_variables_initializer())
       iteration = 1
       for e in range(epochs):
           state = sess.run(initial_state)        
           for x, y in get_batches(train_x, train_y, batch_size):
               feed = {inputs_: x, labels_: y[:, None], keep_prob: 0.5, initial_state: state}
               loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)            
               if iteration%5==0:
                   print("Epoch: {}/{}".format(e, epochs), \
                         "Iteration: {}".format(iteration), \
                         "Train loss: {:.3f}".format(loss))
               if iteration%25==0:
                   val_acc = []
                   val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                   for xv, yv in get_batches(val_x, val_y, batch_size):
                       feed = {inputs_: xv, labels_: yv[:, None], keep_prob: 1, initial_state: val_state}
                       batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                       val_acc.append(batch_acc)
                   print("Val acc: {:.3f}".format(np.mean(val_acc)))
               iteration += 1
       saver.save(sess, "checkpoints/sentiment.ckpt")
```

4. 检验网络
```python
test_acc = []
with tf.Session(graph=graph) as sess:
       saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
       test_state = sess.run(cell.zero_state(batch_size, tf.float32))
       for xt, yt in get_batches(test_x, test_y, batch_size):
           feed = {inputs_: xt, labels_: yt[:, None], keep_prob: 1, initial_state: test_state}
           batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
           test_acc.append(batch_acc)
       print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
```

### Word2Vec
