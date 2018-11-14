---
layout: post
title: "使用Tensorflow实现RNN的简单应用"
tags: [深度学习]
date: 2018-11-14
---

本文使用Tensorflow实现字符到字符的预测（character-wise RNN）

训练文本使用的是[《Anna Karenina》](https://pan.baidu.com/s/1zmmBQQ0t_RmIwlReswxkEg)

建立的模型结构如下图所示

<img src="/img/ctc.png">

RNN建立和训练过程如下：

1. 将文本中的字符编码为整数
```python
import numpy as np
with open('anna.txt', 'r') as f:
       text=f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
```

2. 将编码后的文本转换成输入，每一个输入的batch为一个NxMxI的三维矩阵，其中N为batch size，M为sequence length（RNN处理的序列长度num_steps），I为编码后的字符进行one-hot encode之后的长度（len(vocab)）
```python
import tensorflow as tf
def get_batches(arr, batch_size, num_steps):
       # Get the number of characters per batch and number of batches we can make
       chars_per_batch = batch_size * num_steps
       n_batches = len(arr)//chars_per_batch
       # Keep only enough characters to make full batches
       arr = arr[:n_batches * chars_per_batch].reshape((batch_size, -1))

       for n in range(0, arr.shape[1], num_steps):
           # The features
           x = arr[:, n:n+num_steps]
           # The targets, shifted by one
           y_temp = arr[:, n+1:n+num_steps+1]
           # For the very last batch, use zero to fill in the end of y     
           y = np.zeros(x.shape, dtype=x.dtype)
           y[:,:y_temp.shape[1]] = y_temp

           x_one_hot = tf.one_hot(x, len(vocab))
           y_one_hot = tf.one_hot(y, len(vocab))
           yield x_one_hot, y_one_hot
```

3. 建立RNN的输入层
```python
def build_inputs(batch_size, num_steps):
       # Declare placeholders we'll feed into the graph
       inputs = tf.placeholder(tf.int32, [batch_size, num_steps, len(vocab)], name='inputs')
       targets = tf.placeholder(tf.int32, [batch_size, num_steps, len(vocab)], name='targets')    
       # Keep probability placeholder for drop out layers
       keep_prob = tf.placeholder(tf.float32, name='keep_prob')
       return inputs, targets, keep_prob
```

4. 建立RNN的隐藏层
```python
### create a basic LSTM cell
def build_cell(lstm_size, keep_prob):    
       lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
       drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) #add dropout to the cell
       return drop
### Stack up multiple LSTM layers, for deep learning
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):        
       cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
       initial_state = cell.zero_state(batch_size, tf.float32)
       return cell, initial_state
```

5. 建立RNN的输出层
```python
def build_output(lstm_output, lstm_size):
       ###------------------------------------------------------------------------
       ### Build a softmax layer, return the softmax output and logits.
       ### lstm_output: 3D tensor with shape (batch_size, num_steps, lstm_size)
       ### lstm_size: Size of the LSTM cells
       ###------------------------------------------------------------------------
       # Reshape output so it's a bunch of rows
       # That is, the shape should be batch_size*num_steps rows by lstm_size columns
       x = tf.reshape(lstm_output, [-1, in_size])    
       # Connect the RNN outputs to a softmax layer
       with tf.variable_scope('softmax'): #avoid variable name conflict  
           softmax_w = tf.Variable(tf.truncated_normal((lstm_size, len(vocab)), stddev=0.1))
           softmax_b = tf.Variable(tf.zeros(len(vocab)))       
       # Use softmax to get the probabilities for predicted characters
       logits = tf.matmul(x, softmax_w) + softmax_b
       out = tf.nn.softmax(logits, name='predictions')
       return out, logits
```

6. 构建损失函数和梯度下降优化
```python
### build loss function
def build_loss(logits, targets):
       # reshape to match logits, one row per batch_size per num_steps
       y_reshaped = tf.reshape(targets, logits.get_shape())
       # Softmax cross entropy loss
       loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
       loss = tf.reduce_mean(loss)
       return loss
### build optmizer for training, using gradient clipping to control exploding gradients
def build_optimizer(loss, learning_rate, grad_clip):
       tvars = tf.trainable_variables()
       grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
       train_op = tf.train.AdamOptimizer(learning_rate)
       optimizer = train_op.apply_gradients(zip(grads, tvars))         
       return optimizer
```

7. 建立RNN
