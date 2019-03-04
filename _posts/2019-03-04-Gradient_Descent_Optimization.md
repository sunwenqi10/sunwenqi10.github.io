---
layout: post
title: "梯度下降方法介绍"
tags: [深度学习]
date: 2019-03-04
---

梯度下降公式：$$W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]}$$，$$b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]}$$，具体细节和代码实现参考文章[神经网络介绍](https://sunwenqi10.github.io/blog/2019/02/19/Introduction_to_Neural_Network)

- Batch Gradient Descent:
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

- Stochastic Gradient Descent:
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

### Mini-Batch Gradient Descent

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

The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is the number of examples used to perform one update step.

With a well-tuned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large).
