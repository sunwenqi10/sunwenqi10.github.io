---
layout: post
title: "Fast.ai深度学习第二课"
tags: [深度学习]
date: 2018-09-11
---

**猫狗分类项目**
```python
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "data/dogscats/"
sz=299
arch=resnext50
bs=28

tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=4)
learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5) #ps: dropout parameter(ps=0: no dropout)

learn.fit(1e-2, 1)
learn.precompute=False
learn.fit(1e-2, 2, cycle_len=1)

learn.unfreeze()
lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1)
learn.save('224_all_50')

log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs,y)
```

**Kaggle狗品种分类项目([Data](https://www.kaggle.com/c/dog-breed-identification/data))**：
```python
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
from sklearn import metrics

PATH = "data/dogbreed/"
sz = 224
arch = resnext101_64
bs = 58
label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv))) - 1 # header is not counted (-1)
val_idxs = get_cv_idxs(n) # random 20% data for validation set

def get_data(sz, bs): # sz: image size, bs: batch size
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test', \
                                        val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
    ###The transforms downsize the images to sz. Reading the jpgs and resizing is slow for big images.
    ###Resizing all jpgs to 340 first will save time.
    return data if sz > 300 else data.resize(340, 'tmp')

data = get_data(sz, bs)
learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5) #ps: dropout parameter(ps=0: no dropout)
learn.fit(1e-2, 2)

learn.precompute = False
learn.fit(1e-2, 5, cycle_len=1)
learn.save('224_pre')
learn.load('224_pre')

### Starting training on small images for a few epochs, then switching to bigger images,
### and continuing training is an amazingly effective way to avoid overfitting.
learn.set_data(get_data(299, bs))
learn.freeze()
learn.fit(1e-2, 3, cycle_len=1) # validation loss is much lower than training loss(underfitting)
learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2)
learn.save('299_pre')
learn.load('299_pre')
log_preds, y = learn.TTA() # validation data, (5, 2044, 120), (2044,)
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)
metrics.log_loss(y, probs)

### check if the model can be any better
learn.fit(1e-2, 1, cycle_len=2)
learn.save('299_pre')
log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)
metrics.log_loss(y, probs)

### predict test dataset
log_preds, _ = learn.TTA(is_test=True) # use test dataset rather than validation dataset
probs = np.mean(np.exp(log_preds),0)
df = pd.DataFrame(probs)
df.columns = data.classes
df.insert(0, 'id', [o[5:-4] for o in data.test_ds.fnames])
### write to csv file
SUBM = f'{PATH}/subm/'
os.makedirs(SUBM, exist_ok=True)
df.to_csv(f'{SUBM}subm.gz', compression='gzip', index=False)

### individual prediction
fn = data.val_ds.fnames[0]
trn_tfms, val_tfms = tfms_from_model(arch, sz)
### open_image() returns numpy.ndarray, with shape (image_height, image_width, 3)
img_array = open_image(PATH + fn) # RGB formmat, normalized to range between 0.0~1.0
im = val_tfms(img_array) #(3,sz,sz)
log_preds = learn.predict_array(im[None]) #im[None]: (1,3,sz,sz)
breed = learn.data.classes[np.argmax(log_preds)]
```
