---
layout: post
title: "CART决策树"
tags: [机器学习]
date: 2018-11-03
---

1. 分裂规则
  + 将现有节点的数据分裂成两个子集，计算每个子集的gini index
  + 子集的Gini index: $$gini_{child}=\sum_{i=1}^K p_{ti} \sum_{i' \neq i} p_{ti'}=1-\sum_{i=1}^K p_{ti}^2$$ ， 其中K表示类别个数，$$p_{ti}$$表示分类为i的样本在子集中的比例，gini index可以理解为该子集中的数据被错分成其它类别的期望损失
  + 分裂后的Gini index: $$gini_s= \frac{N_1}{N}gini_{child_1}+\frac{N_2}{N}gini_{child_2}$$ ，其中N为分裂之前的样本数，$$N_1$$和$$N_2$$为分裂之后两个子集的样本数
  + 选取使得$$gini_s$$最小的特征A和分裂点s进行分裂


2. 减少过拟合
  + 设置树的最大深度(max_depth in sklearn.tree.DecisionTreeClassifier)
  + 设置每个叶子节点的最少样本个数(min_samples_leaf in sklearn.tree.DecisionTreeClassifier)
  + 剪枝


3. 样本均衡问题
  + 若样本的类别分布极不均衡，可对每个类i赋予一个权重$$w_i$$, 样本较少的类赋予较大的权重(class_weight in sklearn.tree.DecisionTreeClassifier)，此时算法中所有用到样本类别个数的地方均转换成类别的权重和。例如$$p_{ti}=\frac{w_{i}N_i}{\sum_{i=1}^K w_{i}N_i}$$ ，其中$$N_i$$为在子集中类别为i的样本数; $$gini_s=\frac{weightsum(N_1)}{weightsum(N)}gini_{child_1}+\frac{weightsum(N_2)}{weightsum(N)}gini_{child_2}$$


4. 回归问题
  + 和分类问题相似，只是分裂规则中的$$gini_{child}$$变为了mean squared error，即$$MSE_{child}=\frac{1}{N_{child}}\sum_{i \in child}(y_i-\bar{y}_{child})^2$$
