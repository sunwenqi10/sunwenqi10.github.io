---
layout: post
title: "梯度下降方法介绍"
tags: [深度学习]
date: 2019-03-04
---

### 一、梯度下降

梯度下降公式：$$W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]}$$，$$b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]}$$，具体细节和代码实现参考文章[神经网络介绍](https://sunwenqi10.github.io/blog/2019/02/19/Introduction_to_Neural_Network)

#### (Batch) Gradient Descent:
``` python
### 伪代码
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
      # Forward propagation
      a, caches = forward_propagation(X, parameters)
      # Compute cost.
      cost = compute_cost(a, Y)
      # Backward propagation.
      grads = backward_propagation(a, caches, parameters)
      # Update parameters.
      parameters = update_parameters(parameters, grads)
```

#### Stochastic Gradient Descent:
```python
### 伪代码
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
      for j in range(0, m):
          # Forward propagation
          a, caches = forward_propagation(X[:,j], parameters)
          # Compute cost
          cost = compute_cost(a, Y[:,j])
          # Backward propagation
          grads = backward_propagation(a, caches, parameters)
          # Update parameters.
          parameters = update_parameters(parameters, grads)
```

#### Mini-Batch Gradient Descent

Mini-Batch Gradient Descent介于(Batch) Gradient Descent和Stochastic Gradient Descent之间，可分为两步进行

1. Build mini-batches from the training set (X, Y)
```python
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
       """
       Creates a list of random minibatches from (X, Y)

       Arguments:
       X -- input data, of shape (input size, number of examples)
       Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
       mini_batch_size -- size of the mini-batches, integer

       Returns:
       mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
       """
       np.random.seed(seed)
       m = X.shape[1]   #number of training examples
       mini_batches = []    
       # Step 1: Shuffle (X, Y)
       permutation = list(np.random.permutation(m))
       shuffled_X = X[:, permutation]
       shuffled_Y = Y[:, permutation].reshape((1,m))
       # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
       num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size
       for k in range(0, num_complete_minibatches):
           mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
           mini_batch_Y = shuffled_Y[0,k*mini_batch_size:(k+1)*mini_batch_size].reshape((1,mini_batch_size))
           mini_batch = (mini_batch_X, mini_batch_Y)
           mini_batches.append(mini_batch)   
       # Handling the end case (last mini-batch < mini_batch_size)
       if m % mini_batch_size != 0:
           mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:]
           mini_batch_Y = shuffled_Y[0,num_complete_minibatches*mini_batch_size:].reshape((1,m % mini_batch_size))
           mini_batch = (mini_batch_X, mini_batch_Y)
           mini_batches.append(mini_batch)
       return mini_batches
```

2. Train the network
```python
### 伪代码
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
mini_batches = random_mini_batches(X, Y)
for i in range(0, num_iterations):
       for minibatch in minibatches:
           # Select a minibatch
           (minibatch_X, minibatch_Y) = minibatch
           # Forward propagation
           a, caches = forward_propagation(minibatch_X, parameters)
           # Compute cost
           cost = compute_cost(a, minibatch_Y)
           # Backward propagation
           grads = backward_propagation(a, caches, parameters)
           # Update parameters.
           parameters = update_parameters(parameters, grads)
```

梯度下降、随机梯度下降与mini-batch梯度下降的区别在于进行一次参数更新所使用的训练样本数量；若选取合适的mini_batch_size，mini-batch梯度下降常优于梯度下降和随机梯度下降，特别是当训练数据集很大的情况下。为进一步减少梯度下降过程中不必要的震荡（如下图所示），加快收敛速度，有一些改进的算法可供选择：

![img](/img/kiank_minibatch.png)

#### Momentum

$$ \begin{cases}
v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\
W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}
\end{cases}$$ 与 $$\begin{cases}
v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\
b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}}
\end{cases}$$

#### RMSProp

$$\begin{cases}
s_{dW^{[l]}} = \beta s_{dW^{[l]}} + (1 - \beta) {dW^{[l]}}^2 \\
W^{[l]} = W^{[l]} - \alpha \frac{dW^{[l]}}{\sqrt{s_{dW^{[l]}}} + \varepsilon}
\end{cases}$$与$$\begin{cases}
s_{db^{[l]}} = \beta s_{db^{[l]}} + (1 - \beta) {db^{[l]}}^2 \\
b^{[l]} = b^{[l]} - \alpha \frac{db^{[l]}}{\sqrt{s_{db^{[l]}}} + \varepsilon}
\end{cases}$$

#### Adam

$$\begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) dW^{[l]} \\
v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) {dW^{[l]}}^2 \\
s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_2)^t} \\
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}
\end{cases}$$与$$\begin{cases}
v_{db^{[l]}} = \beta_1 v_{db^{[l]}} + (1 - \beta_1) db^{[l]} \\
v^{corrected}_{db^{[l]}} = \frac{v_{db^{[l]}}}{1 - (\beta_1)^t} \\
s_{db^{[l]}} = \beta_2 s_{db^{[l]}} + (1 - \beta_2) {db^{[l]}}^2 \\
s^{corrected}_{db^{[l]}} = \frac{s_{db^{[l]}}}{1 - (\beta_2)^t} \\
b^{[l]} = b^{[l]} - \alpha \frac{v^{corrected}_{db^{[l]}}}{\sqrt{s^{corrected}_{db^{[l]}}} + \varepsilon}
\end{cases}$$，其中$$t$$表示迭代次数

以上算法中的学习率$$\alpha$$可以是常数，也可以根据训练进度而变化，常用的几种方法有：

#### learning rate decay

有多种形式，例如$$\frac{\alpha_0}{1+decay\_rate*epoch\_number}$$，$$\alpha_0$$为初始学习率

#### cyclic learning rates

有多种形式，例如下图所示

![img](/img/nn3.png)


### 二、正则化
