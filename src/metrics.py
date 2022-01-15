from scipy.stats import wasserstein_distance
from numpy.linalg import norm
from numpy import array, corrcoef


def emd_score(historical, generated):
    return wasserstein_distance(historical, generated)


def acf(x, s, mode='acf'):
    xx = x**2 if mode == 'le' else x
    c0 = corrcoef(x, xx)[0, 1]
    return array([c0] + [corrcoef(x[:-i], xx[i:])[0, 1] for i in range(1, s)])


def dependence_score(historical, generated_set, s, mode, func=None):
    if func is None:
        def func(x): return x

    acf_hist = acf(func(historical), s, mode)
    acf_gen = array([acf(func(gen), s, mode) for gen in generated_set])
    return norm(acf_hist - acf_gen.mean(axis=0))


def acf_score(historical, generated_set, s, func=None):
    return dependence_score(historical, generated_set, s, 'acf', func)

def le_score(historical, generated_set, s):
    return dependence_score(historical, generated_set, s, 'le')
