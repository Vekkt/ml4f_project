from scipy.stats import wasserstein_distance
from numpy.linalg import norm
from numpy import correlate, array


def emd(historical, generated):
    return wasserstein_distance(historical, generated)


def acf(r, mode):
    if mode == 'le':
        corr = correlate(r ** 2, r, mode='full')
    elif mode == 'acf':
        corr = correlate(r, r, mode='full')
    else:
        raise ValueError("Unknown specified mode for the autocorrelation function.")
    return corr[corr.size // 2:]


def dependence_score(historical, generated_set, mode, func=None):
    if func is None:
        def func(x): return x

    acf_hist = acf(func(historical), mode)
    acf_gen = array([acf(func(gen), mode) for gen in generated_set])
    return norm(acf_hist - acf_gen.mean(axis=1))


def acf_score(historical, generated_set, func=None):
    return dependence_score(historical, generated_set, 'acf', func)


def le_score(historical, generated_set):
    return dependence_score(historical, generated_set, 'le')
