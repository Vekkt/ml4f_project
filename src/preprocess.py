from bdb import Breakpoint
import numpy as np
from scipy.stats import kurtosis, norm, normaltest
from scipy.optimize import fmin
from scipy.special import lambertw
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt


def delta_gmm(x, gamma_0=3):
  delta_0 = 1 / 66 * (np.sqrt(66 * gamma_0 - 162) - 6)

  def f(delta):
    u_delta = w_delta(x, delta)
    gamma = kurtosis(u_delta, fisher=False)
    return np.abs(gamma - gamma_0)

  res = fmin(f, delta_0, disp=False)

  return res[0]


def w_delta(z, delta):
  return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)


def igmm(y, tol=1e-6, n_max=1000):
  tau_prev = np.array([0, 0, 0])
  gamma = kurtosis(y, fisher=False)
  delta = 1 / 66 * (np.sqrt(66 * gamma - 162) - 6)
  tau = np.array([np.median(y), np.std(y) * np.power(1. - 2. * delta, 0.75), delta])
  k = 0
  
  while np.linalg.norm(tau - tau_prev) > tol and k < n_max:
    tau_prev = copy(tau)
    z = (y - tau[0]) / tau[1]
    tau[2] = delta_gmm(z) #at k+1
    u = w_delta(z, tau[2])
    x = u * tau[1] + tau[0]
    tau[0], tau[1] = np.mean(x), np.std(x)
    k = k + 1
    print('', end='\r')

  if k > n_max:
    raise RuntimeError(f"IGMM did not converge after {n_max} iterations")
  print(np.linalg.norm(tau - tau_prev), k)
  return tau

def inv_transf(u, tau):
     return tau[0] + tau[1] * u * np.exp(0.5 * tau[2] * u **2)


class Gaussianize():
     def __init__(self):
          self.tau = np.array([0, 0, 0])
     


     def fit(self, logret: np.array):
          #standartize returns 
          ret_mean, ret_std  = logret.mean(), logret.std()
          logret_norm = (logret - ret_mean) / ret_std 
          
          self.logret_coefs = [ret_mean, ret_std]
          self.tau = igmm(logret_norm)
     
     def transform(self, logret: np.array):
          #standartize returns
          logret_norm = (logret - self.logret_coefs[0]) / self.logret_coefs[1] 
          y = logret_norm #so that it matches the paper notation
          #check if the returns are fitted
          if all(self.tau) == 0:
               raise ValueError('Fit the data first')

          z = (y - self.tau[0]) / self.tau[1] #from the paper
          #u is the standartized version of X, where X is the gaussianized y
          u = w_delta(z, self.tau[2]) #from the paper
          return u

     def inverse_transform(self, u: np.array):
          """Here u is the fake data with (theoretically 0 mean and std=1)
          Since we used u to generate fake distribution."""
          if all(self.tau) == 0:
               raise ValueError('Fit the data first')

          y = inv_transf(u, self.tau)

          return y * self.logret_coefs[1] +  self.logret_coefs[0]



def main():
     raw_data = pd.read_csv('raw_data.csv')
     logret = np.diff(np.log(raw_data['Close']))
     gauss = Gaussianize()
     gauss.fit(logret)
     u = gauss.transform(logret)


     np.random.seed(0)
     #plt.hist(norm.rvs(100), bins=50, label='norm', alpha=0.1)
     loc, scale = norm.fit(u)
     print(f'loc: {loc:.6f}, scale: {scale:.6f}')
     print(f'kurtosis: {kurtosis(u, fisher=False):.6f}')
     print(normaltest(u))
     xmin, xmax = u.min(), u.max()
     x = np.linspace(xmin, xmax)
     plt.title('Inverse Lambert W Transform')
     plt.ylabel('Density')
     plt.plot(x, norm.pdf(x, loc, scale), label='Theoretical')
     plt.hist(u, alpha=0.5, bins=50, label='Tranform', density=True)
     #plt.hist(logret_norm, alpha=0.5, bins=50, label='log_ret', density=True )
     plt.legend()
     plt.pause(5)

     inverse_transformed = gauss.inverse_transform(u)
     
     plt.hist(logret, bins=50, alpha=0.5, label='real')
     plt.hist(inverse_transformed, bins=50, alpha=0.5, label='inverse')
     plt.legend()
     plt.show()







if __name__ == '__main__':
     main()

