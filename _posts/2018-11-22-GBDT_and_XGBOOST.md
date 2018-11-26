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

针对$$w_{mkj}$$的计算，有$$w_{mkj(j=1...J_{mk},k=1...K)}=argmin_{w_{kj(j=1...J_{mk},k=1...K)}}\sum_{i=1}^{N}L(y_i, ...,   F_{m-1,k}(x_i)+\sum_{j=1}^{J_{mk}}w_{kj}I({x}\in{R_{mkj}}), ...)$$   
