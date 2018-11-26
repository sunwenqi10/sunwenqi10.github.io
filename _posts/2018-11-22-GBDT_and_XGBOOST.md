---
layout: post
title: "GBDT和XGBOOST算法原理"
tags: [机器学习]
date: 2018-11-22
---

### GBDT

本文以多分类问题为例介绍GBDT的算法，针对多分类问题，每次迭代都需要生成K个树（K为分类的个数），记为$$F_{mk}(x)$$，其中m为迭代次数，k为分类。

针对每个训练样本，使用的损失函数通常为$$L(y_i, F_{m1}(x_i), ..., F_{mK}(x_i))=-\sum_{k=1}^{K}I({y_i}=k)ln[p_{mk}(x_i)]=-\sum_{k=1}^{K}I({y_i}=k)ln(\frac{e^{F_{mk}(x_i)}}{\sum_{l=1}^{K}e^{F_{ml}(x_i)}})$$，此时损失函数的梯度可以表示为$$g_{mki}=-\frac{\partial{L(y_i, F_{m1}(x_i), ..., F_{mK}(x_i))}}{\partial{F_{mk}(x_i)}}=I({y_i}=k)-p_{mk}(x_i)=I({y_i}=k)-\frac{e^{F_{mk}(x_i)}}{\sum_{l=1}^{K}e^{F_{ml}(x_i)}}$$。

GBDT算法的流程如下所示：
1. for k=1 to K: Initialize $$F_{0k}(x)=0$$
2. for m=1 to M:
  + for k=1 to K: compute $$g_{m-1,ki}$$ for each sample $$(x_i, y_i)$$
  + for k=1 to K: build up regression tree $$R_{mkj}$$(j=1 to J<sub>mk</sub> refer to the leaf nodes) from training samples $$(x_i, g_{m-1,ki})_{i=1,...,N}$$
  + for k=1 to K: compute leaf weights $$w_{mkj}$$ for j=1 to J<sub>mk</sub>
  + for k=1 to K: $$F_{mk}(x)=F_{m-1,k}(x)+\eta*\sum_{j=1}^{J_{mk}}w_{mkj}I({x}\in{R_{mkj}})$$, $$\eta$$为学习率

针对$$w_{mkj}$$的计算，有$$w_{mkj(j=1...J_{mk},k=1...K)}=argmin_{w_{kj(j=1...J_{mk},k=1...K)}}\sum_{i=1}^{N}L(y_i, ...,   F_{m-1,k}(x_i)+\sum_{j=1}^{J_{mk}}w_{kj}I({x_i}\in{R_{mkj}}), ...)$$

为了求得w的值，使上述公式的一阶导数为0，问题转化为F(x)=0类型的问题，利用Newton-Raphson公式（在这个问题中将初始值设为0，只进行一步迭代，并且Hessian矩阵只取对角线上的值），记$$L_i=L(y_i, ...,   F_{m-1,k}(x_i)+\sum_{j=1}^{J_{mk}}w_{kj}I({x_i}\in{R_{mkj}}), ...)$$，有$$w_{mkj}=-\frac{\sum_{i=1}^N\partial{L_i}/\partial{w_{kj}}}{\sum_{i=1}^N\partial^2{L_i}/\partial{w_{kj}^2}}=\frac{\sum_{i=1}^{N}I({x_i}\in{R_{mkj}})[I({y_i}=k)-p_{m-1,k}(x_i)]}{\sum_{i=1}^{N}I^2({x_i}\in{R_{mkj}})p_{m-1,k}(x_i)[1-p_{m-1,k}(x_i)]}=\frac{\sum_{x_i\in{R_{mkj}}}g_{m-1,ki}}{\sum_{x_i\in{R_{mkj}}}\lvertg_{m-1,ki}\rvert(1-\lvertg_{m-1,ki}\rvert)}$$

参考文献

[1] Friedman, Jerome H. Greedy function approximation: A gradient boosting machine. Ann. Statist. 29 (2001), 1189--1232.
