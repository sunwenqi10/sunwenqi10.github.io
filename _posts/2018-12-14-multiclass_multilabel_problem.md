---
layout: post
title: "机器学习应用示例：多类别多标签问题"
tags: [机器学习]
date: 2018-12-14
---

本文使用的数据（[下载地址](https://pan.baidu.com/s/1w8MI70oAwK_3knEjdzW1aQ)）是一个多类别多标签分类问题，具体介绍和问题描述参考[此链接](https://www.drivendata.org/competitions/4/box-plots-for-education/page/15)

1. 拆分训练集和测试集
```python
def multilabel_sample(y, size=1000, min_count=5, seed=None):
    """ Takes a matrix of binary labels `y` and returns
        the indices for a sample of size `size` if
        `size` > 1 or `size` * len(y) if size =< 1.
        The sample is guaranteed to have > `min_count` of
        each label.
    """
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).any():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')
    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')
    if size <= 1:
        size = np.floor(y.shape[0] * size)
    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))
    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])
    sample_idxs = np.array([], dtype=choices.dtype)
    # first, guarantee > min_count of each label
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])
    sample_idxs = np.unique(sample_idxs)
    # now that we have at least min_count of each, we can just random sample
    sample_count = int(size - sample_idxs.shape[0])
    # get sample_count indices from remaining choices
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices,
                                   size=sample_count,
                                   replace=False)
    return np.concatenate([sample_idxs, remaining_sampled])   
### Takes a features matrix X and a label matrix Y
### Returns (X_train, X_test, Y_train, Y_test) where all classes in Y are represented at least min_count times.     
def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):       
       index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
       test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
       test_set_mask = index.isin(test_set_idxs)
       train_set_mask = ~test_set_mask
       return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])
```
