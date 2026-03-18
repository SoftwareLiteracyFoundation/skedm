skedm - Empirical Dynamic Modeling with scikit-learn
============================================================

**skedm** provides Empirical Dynamic Modeling [EDM](https://en.wikipedia.org/wiki/Empirical_dynamic_modeling) tools within the [scikit-learn](https://scikit-learn.org) ecosystem.

#### Demo
Here we demonstrate `skedm.Simplex` on a multivariate time series prediction task.

A five variable nonlinear (chaotic) dynamical system is represented in `data/Lorenz5D.csv` from the Lorenz'96 model. 

```python
from pandas import read_csv
df = read_csv('Lorenz5D.csv') # see ./skedm/skedm/data/
df.head(3)
    Time      V1      V2      V3      V4      V5
0  10.00  2.4873  1.0490  3.4093  8.6502 -2.4232
1  10.05  3.5108  2.2832  4.0464  7.8964 -2.1931
2  10.10  4.1666  3.7791  4.7456  6.8123 -1.8866
```

The task is to predict variable `V2` from the other four. To specify a multivariate embedding instead of a time-delay embedding we set the parameter `embedded=True`. The embedding is constructed from `columns`. We define a training set (library) `lib=[1,300]` and test set (prediction) `pred=[601,800]`. 

```python
from skedm import Simplex
columns, target = ['V1', 'V3', 'V4', 'V5'], 'V2'
lib, pred = [1,300], [601,800]

smp = Simplex(columns=columns, target=target, Tp=0, embedded=True, lib=lib, pred=pred)
smp.fit(df)
smp.score(df, df[target])
0.9496
```

`Simplex.predict()` creates a `Projection_` attribute, a DataFrame of the `Observations` (`target`) along with `Predictions`

```python
smp.Projection_.head(3)
    Time  Observations  Predictions  Pred_Variance
0  40.00        7.5310     7.561277       0.191294
1  40.05        6.7717     6.760938       0.242625
2  40.10        5.7151     5.648007       0.352205
```

```python
from skedm.aux_func import PlotObsPred
PlotObsPred(smp.Projection_,E=smp._E,Tp=smp.Tp)
```
<img width="539" height="442" alt="skedm_simplex_demo" src="https://github.com/user-attachments/assets/1716e13f-7e3a-49fc-81ee-aafbd265da72" />
