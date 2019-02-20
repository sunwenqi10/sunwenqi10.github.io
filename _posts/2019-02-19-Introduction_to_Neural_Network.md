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

### 公式

#### 1. Forward Propagation

线性部分：$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$$（注：$$A^{[0]}=X$$）

非线性部分：$$A^{[l]}=g(Z^{[l]})$$（$$g$$为激活函数）
+ 本文隐藏层的激活函数使用relu，可减轻梯度消失问题
+ 若为二分类问题，输出层的激活函数使用sigmoid；若为回归问题，输出层不使用激活函数，即$$A^{[L]}=Z^{[L]}$$

#### 2. Loss Function

若为回归问题，损失函数可写为$$\mathcal{J}=\frac{1}{2m}\sum\limits_{i = 1}^{m}(a^{[L] (i)}-y^{(i)})^2$$，其中$$a^{[L] (i)}$$为第$$i$$个样本的预测值（即$$A^{[L]}$$的$$i$$列），$$y^{(i)}$$为第$$i$$个样本的真实值

若为二分类问题，损失函数可写为$$\mathcal{J}=-\frac{1}{m} \sum\limits_{i = 1}^{m} [y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)]$$

#### 3. Backward Propagation

记$$dA^{[l]}=\frac{\partial \mathcal{J} }{\partial A^{[l]}}$$，则可推出以下公式：
+ (1) $$dZ^{[l]}=\frac{\partial \mathcal{J} }{\partial Z^{[l]}}=dA^{[l]}* g'(Z^{[l]})$$，其中$$g'$$表示激活函数的导数
+ (2) $$dW^{[l]} = \frac{\partial \mathcal{J} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T}$$，其中$$A^{[l-1] T}$$表示$$A^{[l-1]$$的转置
+ (3) $$db^{[l]} = \frac{\partial \mathcal{J} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}$$，其中$$dZ^{[l](i)}$$为矩阵$$dZ^{[l]}$$的第$$i$$列
+ (4) $$dA^{[l-1]} = \frac{\partial \mathcal{J} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]}$$，其中$$W^{[l] T}$$表示$$W^{[l]}$$的转置

#### 4. Update Parameters

$$W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]}$$，$$b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]}$$，$$\alpha$$为学习率

### 代码

#### Initialize parameters
```python
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)   # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))    
    return parameters
```

#### Forward Propagation
```python
def sigmoid(Z): return 1/(1+np.exp(-Z))
def relu(Z): return np.maximum(0,Z)
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" or "none"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A_prev)+b
    linear_cache = (A_prev, W, b)
    activation_cache = Z
    A = sigmoid(Z) if activation=="sigmoid" else relu(Z) if activation=="relu" else Z
    cache = (linear_cache, activation_cache)
    return A, cache
def L_model_forward(X, parameters, type):
    """
    Implement forward propagation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    type -- problem type, stored as a text string: "binary classification" or "regression"

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2   # number of layers in the neural network(excluding the input layer)
    ### hidden layer
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    ### output layer
    if type=="regression":
        AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'none')
    else:
        AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)            
    return AL, caches
```

#### Loss Function
```python
def compute_cost(AL, Y, type):
    """
    Arguments:
    AL -- last post-activation value, shape (1, number of examples)
    Y -- true vector, shape (1, number of examples)
    type -- problem type, stored as a text string: "binary classification" or "regression"

    Returns:
    cost -- cross-entropy loss for classification and mean squared error for regression
    """

    m = Y.shape[1] #number of examples
    if type=="regression":
        cost = np.sum(np.power(AL-Y,2))/(2*m)
    else:
        cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
    cost = np.squeeze(cost)  # To make sure cost's shape is what expected (e.g., this turns [[10]] into 10)
    return cost
```

#### Backward Propagation
```python
```
