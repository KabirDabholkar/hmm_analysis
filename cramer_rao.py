import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm

mu = 0.5
sigma = 1

estimator_variance = []
mean_log_likelihood = []
mean_log_likelihood2 = []
sample_sizes = np.logspace(1.2,2.5,5).round().astype(int)

test_samples = 10000

test_X = np.random.normal(mu,sigma,size=(test_samples))
for samples in sample_sizes:
    # samples = 5
    repeats = 10000
    X = np.random.normal(mu,sigma,size=(samples,repeats))
    sample_mean = X.mean(axis=0,keepdims=True)
    mean_log_likelihood += [
        # - np.log(np.sqrt(2*np.pi)*sigma) - np.mean((test_X-sample_mean)**2/(2*sigma**2))
        (np.log(norm.pdf(test_X,sample_mean,sigma)).mean())
    ]
    mean_log_likelihood2 += [
        - np.log(np.sqrt(2*np.pi)*sigma) - np.mean((test_X-sample_mean)**2/(2*sigma**2))
        # (np.log(norm.pdf(test_X,sample_mean,sigma)).mean())
    ]
    estimator_variance += [sample_mean.std()**2]
    # print('Mean loglikelihood (numerical,analytical):',mean_log_likelihood,-np.log(2*sigma)-0.5*(1+1/samples+mu**2/sigma**2))
    # print('Estimator Variance (numerical,analytical):',sample_mean.std()**2,sigma**2/samples)


# plt.plot(sample_sizes,mean_log_likelihood,color='C0')
fig,axs=plt.subplots(2,1)
ax = axs[0]
ax.plot(sample_sizes,sigma**2/sample_sizes,color='C0',ls='dashed')
ax.plot(sample_sizes,estimator_variance,color='C1')


ax = axs[1]
ax.plot(sample_sizes, - np.log(np.sqrt(2*np.pi)*sigma)-0.5*(1+1/sample_sizes),color='C0',ls='dashed')
ax.plot(sample_sizes,mean_log_likelihood,color='C1')
# ax.plot(sample_sizes,mean_log_likelihood2,color='C2',ls='dashed')
fig.savefig('plots/test_plots/gaussian_estimator_stats.png')
plt.close()

# plt.figure()
# plt.plot(sample_sizes,sigma**2/sample_sizes,color='C0',ls='dashed')
# plt.plot(sample_sizes,estimator_variance,color='C1')
# plt.savefig('plots/test_plots/gaussian_estimator_variance.png')
# plt.close()

A = np.zeros()
np.arange(10) % 5