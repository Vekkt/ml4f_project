from scipy.stats import wasserstein_distance
from numpy.linalg import norm
from numpy import correlate, vectorize

def emd(historical, generated):
    return wasserstein_distance(historical, generated)


def acf(r, mode):
    if mode is 'le':
        corr = correlate(r ** 2, r, mode='full')
    elif mode is 'acf':
        corr = correlate(r, r, mode='full')
    else:
        raise ValueError("Unknown specified mode for the autocorrelation function.")
    return corr[corr.size/2:]


def dependence_score(historical, generated_set, mode, func=None):
    if func is None:
        func = lambda x: x
    vect_acf = vectorize(lambda r: acf(func(r), mode))
    
    acf_hist = acf(func(historical), mode)
    acf_gen = vect_acf(generated_set).mean
    return norm(acf_hist, acf_gen)


def acf_score(historical, generated_set, func=None):
    return dependence_score(historical, generated_set, 'acf', func)


def le_score(historical, generated_set):
    return dependence_score(historical, generated_set, 'le')



from numpy import array
hist = array([1, 2, 3])
gen = array([array([1, 3, 5]), array([1, 2, 3]), array([1, 2, 3])])
print(emd(hist, gen[0]))
print(acf(hist, mode='acf'))
print(acf(hist, mode='le'))
print(acf_score(hist, gen))
print(le_score(hist, gen))
