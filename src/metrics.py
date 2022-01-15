from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acf
from scipy.stats import pearsonr
import numpy as np

def emd_score(historical, generated):
    return wasserstein_distance(historical, generated)


def acf_(x, s, mode='acf'):
     xx = x**2 if mode == 'le' else x
     c0 = pearsonr(x, xx)[0]
     return np.array([c0] + [pearsonr(x[:-i], xx[i:])[0] for i in range(1, s)])


def dependence_score(historical, generated_set, s, mode, func=None):
     if func is None:
          def func(x): return x

     acf_hist = acf_(func(historical), s, mode)
     acf_gen = np.array([acf_(func(gen), s, mode) for gen in generated_set])
     return np.linalg.norm(acf_hist - acf_gen.mean(axis=0))


def acf_score(historical, generated_set, s, func=None):
     if func is None:
          def func(x): return x

     acfs = [acf(func(generated_set[i, :]), nlags=s, fft=False)[1:] for i in range(generated_set.shape[0])]
     mean_acf = np.array(acfs).mean(axis=0)
     return np.linalg.norm(acf(func(historical), nlags=s, fft=False)[1:] - mean_acf)
    

def le_score(historical, generated_set, s):
    return dependence_score(historical, generated_set, s, 'le')
