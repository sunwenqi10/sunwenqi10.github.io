---
layout: post
title: "使用R语言进行时间序列分析—基础"
tags: [时间序列]
date: 2019-03-06
---

时间序列模型$$Y_t=m_t+s_t+X_t$$，其中$$m_t$$为趋势项，$$s_t$$为季节项（假设周期为$$d$$，则$$s_t=s_{t+d}$$并且$$\sum_{j=1}^ds_j=0$$），$$X_t$$为平稳项（统计特性不随时间变化而改变）

对于趋势项有两种方式处理：

1. 估计趋势并从原序列中去除，趋势的估计方法主要有以下几种：

   - 滑动平均：可理解为Kernel Regression的一种特殊形式（Nadaraya–Watson estimator），假设滑动窗口为$$2h$$，则$$\hat{m}_t=\frac{\sum_{i=1}^TI(\lvert{i-t}\rvert\leq{h})Y_i}{\sum_{i=1}^TI(\lvert{i-t}\rvert\leq{h})}$$

   - 线性回归：$$\hat{m}_t=\beta_0+\sum_{k=1}^p\beta_kt^k$$($$p$$通常取1或2)

   - Local Polynomial Regression：取最近的k个点进行加权线性回归，假设窗口为$$2h$$，核函数取为$$D(x)=\begin{cases}(1-\lvert{x}\rvert^3)^3, \text{if } \lvert{x}\rvert\leq{1} \\ 0, \text{otherwise}\end{cases}$$，则求解$$argmin_{\beta_k(t),k=0,1,...,p}\sum_{i=1}^TD(\frac{i-t}{h})[Y_i-\sum_{k=0}^p\beta_k(t)i^k]^2$$($$p$$通常取1或2)，$$\hat{m}_t=\beta_0(t)+\sum_{k=1}^p\beta_k(t)t^k$$

   - Splines Regression：假设选中的knots为$$t_1<t_2<...<t_K$$，以cubic spline为例，它的基函数可取为$$B_1(t)=1,B_2(t)=t,B_3(t)=t^2,B_4(t)=t^3,B_5(t)=(t-t_1)^3_+,...,B_{K+4}(t)=(t-t_K)^3_+$$，$$\hat{m}_t=\sum_{j=1}^{K+4}\beta_jB_j(t)$$；和上述方式等价但能有效减少计算量的方式是cubic B-spline，记

2. 将相邻数据相减从而直接去除趋势，即$$\hat{Y}_t=Y_t-Y_{t-1}$$
