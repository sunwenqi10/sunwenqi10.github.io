---
layout: post
title: "GBDT和XGBOOST算法原理"
tags: [机器学习]
date: 2018-11-22
---

本文以多分类问题为例介绍GBDT的算法，针对多分类问题，每次迭代都需要生成K个树（K为分类的个数），记为$$F_{mk}(x)$$，其中m为迭代次数，k为分类。

针对每个训练样本，使用的损失函数通常为$$L(y_i, F_{m1}(x_i), ..., F_{mK}(x_i))=\sum_{k=1}^{K}I({y_i}\in{k})lnp_k(x_i)=$$
