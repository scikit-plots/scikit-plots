## Changelog

### DeprecationWarning: scipy.interp is deprecated and will be removed in SciPy 2.0.0, use numpy.interp instead

- https://github.com/scikit-learn/scikit-learn/pull/21911
```
from scipy import interp change to np.interp

$ git grep "interp("
scikitplot/metrics.py:        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
scikitplot/metrics.py:            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
scikitplot/plotters.py:        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
```
