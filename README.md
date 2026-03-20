skedm - scikit-learn compatible Empirical Dynamic Modeling
==========================================================

**skedm** is a [scikit-learn](https://scikit-learn.org) compatible implementation of [Empirical Dynamic Modeling (EDM)](https://en.wikipedia.org/wiki/Empirical_dynamic_modeling). Core code is based on the [pyEDM](https://pypi.org/project/pyEDM/) package with introduction to EDM and examples in [pyEDM Docs](https://sugiharalab.github.io/EDM_Documentation/).

### Demos

Consider a five variable nonlinear (chaotic) dynamical system of the Lorenz'96 model as represented in `data/Lorenz5D.csv`:

```python
from pandas import read_csv
df = read_csv('Lorenz5D.csv') # see ./skedm/skedm/data/
df.head(3)
    Time      V1      V2      V3      V4      V5
0  10.00  2.4873  1.0490  3.4093  8.6502 -2.4232
1  10.05  3.5108  2.2832  4.0464  7.8964 -2.1931
2  10.10  4.1666  3.7791  4.7456  6.8123 -1.8866
```

We wish to predict variable (`V2`) without `V2` itself but from the other four variables, a four dimensional embedding of `[V1, V3, V4, V5]`.

---

#### SMap Projection

Sequentially locally weighted global linear maps (s-map) facilitate prediction, quantification of intervariable dependencies (Jacobians) as well as scale of nonlinearity of multivariate or time-lagged dynamical systems ([Sugihara 1994](https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1994.0106)). Here we demonstrate `skedm.SMap` in multivariate time series prediction and variable interaction. To specify a multivariate embedding instead of a time-delay embedding we set the parameter `embedded=True`. The embedding is constructed from the specified `columns` and we use a nearest neighbor localization parameter `theta=8` characterizing local neighbor scale in the embedding.

```python
from skedm import SMap
columns, target = ['V1', 'V3', 'V4', 'V5'], 'V2'

smap = SMap(columns=columns, target=target, Tp=0., theta=8., embedded=True)
smap.fit(df)
smap.score(df, df['V2'])
0.9443
```

SMap predictions are stored in the `Projection_` attribute, a DataFrame:
```
smap.Projection_.head(3)
    Time  Observations  Predictions  Pred_Variance
0  10.00        1.0490     1.123976       2.294987
1  10.05        2.2832     2.339546       2.653860
2  10.10        3.7791     3.805514       3.460904
```

with SMap coefficients representing the time-varying, state-dependent interactions (Jacobians) in the `Coefficients_` attribute:
```
smap.Coefficients_.head(3)
    Time        C0   ∂V2/∂V1   ∂V2/∂V3   ∂V2/∂V4   ∂V2/∂V5
0  10.00  4.071493  0.471095  0.272780 -0.492671  0.325006
1  10.05  3.652508  0.571183  0.368843 -0.505170  0.374690
2  10.10  3.262128  0.715885  0.428880 -0.550513  0.383998
```

```python
from skedm.aux_func import PlotObsPred, PlotCoeff
PlotObsPred(smap.Projection_, E=smap._E, Tp=smap.Tp)
PlotCoeff(smap.Coefficients_, "Lorenz 5D", E=smap._E, Tp=smap.Tp)
```

#### Comparison to GaussianProcessRegressor

Estimation of intervariable Jacobians are commonly performed with a Gaussian process regressor applied to time differenced evolution of system variables. For example

```python
X,y = df[columns], df[target]

from pandas import concat, DataFrame
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
gpr = GaussianProcessRegressor(kernel=RBF(), alpha=0.1)
gpr.fit(X, y)

partialDeriv = {}
epsilon = 1e-4
for i,col in enumerate(columns):
    X_epsilon = X.copy()
    X_epsilon.iloc[:, i] += epsilon
    y_pred = gpr.predict(X)
    y_pred_epsilon = gpr.predict(X_epsilon)
    partialDeriv[f'∂{target}/∂{col}'] = (y_pred_epsilon - y_pred) / epsilon

Derivatives = concat([df, DataFrame(partialDeriv)], axis=1)
```

Comparison of SMap and Gaussian process estimates of the partials show both provide reasonable results however, close examination reveals SMap estimates to be more accurate than GP. Further, GP requires both regressing the dynamics and forming a suitable estimate of the derivative based on an ad-hoc perturbation (`epsilon`). In SMap the scale parameter `theta` can be optimally determined to best represent the underlying dynamics in the embedding naturally yielding scale appropriate Jacobians and singular values of the eigendecompositions useful for interaction and change detection. 


---

#### Simplex Projection

Here we demonstrate `skedm.Simplex` in multivariate time series prediction ([Sugihara 1990](https://www.nature.com/articles/344734a0),[Deyle 2011](https://doi.org/10.1371/journal.pone.0018295)). The multivariate embedding is constructed from the specified `columns` with `embedded=True`and we define a training set (library) `lib=[1,300]` and out of sample test set (prediction) `pred=[601,800]`.

```python
from skedm import Simplex
columns, target = ['V1', 'V3', 'V4', 'V5'], 'V2'
lib, pred = [1,300], [601,800]

smp = Simplex(columns=columns, target=target, Tp=0, embedded=True, lib=lib, pred=pred)
smp.fit(df)
smp.score(df, df[target])
0.9496
```

`Simplex.predict()` creates a `Projection_` attribute, a DataFrame of `Observations` (`target`) along with `Predictions`

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
