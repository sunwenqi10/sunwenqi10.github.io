---
layout: post
title: "机器学习应用示例：多类别多标签问题"
tags: [机器学习]
date: 2018-12-14
---

本文使用的数据（[下载地址](https://pan.baidu.com/s/1w8MI70oAwK_3knEjdzW1aQ)）是一个多类别多标签分类问题，具体介绍和问题描述参考[此链接](https://www.drivendata.org/competitions/4/box-plots-for-education/page/15)

1. 拆分训练集和测试集
```python
import numpy as np
import pandas as pd
from warnings import warn
### Takes a label matrix 'y' and returns the indices for a sample with size
### 'size' if 'size' > 1 or 'size' * len(y) if 'size' <= 1.
### The sample is guaranteed to have > 'min_count' of each label.
def multilabel_sample(y, size=1000, min_count=5, seed=None):
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
           size = y.shape[1] * min_count #size should be at least this value for having >min_count of each label
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
       remaining_sampled = rng.choice(remaining_choices, size=sample_count, replace=False)
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

2. 定义loss metric(Logarithmic Loss metric)
```python
### 数据共对应9个标签，每个标签又有不同的类别个数
LABEL_INDICES = [range(0, 37), range(37, 48), range(48, 51), range(51, 76), range(76, 79), \
                    range(79, 82), range(82, 87), range(87, 96), range(96, 104)]
### Logarithmic Loss metric
### predicted, actual: 2D numpy array
def multi_multi_log_loss(predicted, actual, label_column_indices=LABEL_INDICES, eps=1e-15):
       label_scores = np.ones(len(label_column_indices), dtype=np.float64)
       # calculate log loss for each set of columns that belong to a label
       for k, this_label_indices in enumerate(label_column_indices):
           # get just the columns for this label
           preds_k = predicted[:, this_label_indices].astype(np.float64)
           # normalize so probabilities sum to one (unless sum is zero, then we clip)
           preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)
           actual_k = actual[:, this_label_indices]
           # shrink predictions
           y_hats = np.clip(preds_k, eps, 1 - eps)
           sum_logs = np.sum(actual_k * np.log(y_hats))
           label_scores[k] = (-1.0 / actual.shape[0]) * sum_logs
       return np.average(label_scores)        
```

3. 仅使用数值特征建模
```python
### 读取数据
df = pd.read_csv('TrainingData.csv', index_col=0)
NUMERIC_COLUMNS = ['FTE', 'Total']
LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', \
             'Object_Type', 'Pre_K', 'Operating_Status']
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)
label_dummies = pd.get_dummies(df[LABELS], prefix_sep='__')
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only, label_dummies, \
                                                                  size=0.2, seed=123)
### 使用Logistic分类
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)
print("Test Logloss: {}".format(multi_multi_log_loss(predictions, y_test.values)))

### 对文本数据使用bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
# converts all text in each row of data_frame to single vector
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):   
    to_drop = set(to_drop) & set(data_frame.columns.tolist()) #Drop non-text columns that are in the df
    text_data = data_frame.drop(to_drop, axis=1)
    text_data.fillna('', inplace=True)
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: ' '.join(x), axis=1)
text_vector = combine_text_columns(df)
# create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'  #(?=re)表示当re也匹配成功时输出(前面的部分
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
vec_alphanumeric.fit_transform(text_vector)
vec_alphanumeric.get_feature_names()
```
