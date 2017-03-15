import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
def make_draws(dist, params, size=200):
  """
  Draw samples of random variables from a specified distribution, with given
  parameters return these in an array.

  INPUT:
  dist: (Scipy.stats distribution object) Distribution with a .rvs method
  params: (dict) Parameters to define the distribution dist.
                e.g. if dist = scipy.stats.binom then params could be:
                {'n': 100, 'p': 0.25}
  size: (int) Number of examples to draw

  OUTPUT:
  (Numpy array) Sample of random variables
  """
  return dist(**params).rvs(size)

# Generate draws from the Binomial Distribution, using Scipy's binom object.

# binomial_samp = make_draws(scs.binom, {'n': 100, 'p':0.25}, size=200)
#
# poisson_samp = make_draws(scs.poisson, {'mu': 25})
#
# exponential_samp = make_draws(scs.expon, {'loc':0, 'scale':1})
#
# geom_samp = make_draws(scs.geom, {'p': .25})
#
# uniform_samp = make_draws(scs.uniform, {})


def plot_means(repeat,dist,params,size,show=False):
    res = []
    for i in xrange(repeat):
        res.append(np.mean(dist(**params).rvs(size)))
    plt.hist(res)
    if show:
        plt.show()



# binomial_samp = plot_means(5000,scs.binom, {'n': 100, 'p':0.25}, size=200)
# poisson_samp = plot_means(5000,scs.poisson, {'mu':25}, size=200)
# exponential_samp = plot_means(5000,scs.expon, {'loc':0, 'scale':1}, size=200)
# geom_samp = plot_means(5000,scs.geom, {'p': .25}, size=200)
# uniform_samp = plot_means(5000,scs.uniform, {}, size=200)

'''
The range of sampling distribution grow wider as the sample size decrease due to higher variance

'''


def plot_max(repeat,dist,params,size,show=False):
    res = []
    for i in xrange(repeat):
        res.append(np.max(dist(**params).rvs(size)))
    plt.hist(res)
    if show:
        plt.show()

filepath = "/Users/twilightidol/Downloads/data2/lunch_hour.txt"
lunch = np.loadtxt(filepath, delimiter=',')

mu = np.mean(lunch)
stan_error = np.std(lunch)/(len(lunch)**.5)
#95% confident interval is
z=scs.norm.ppf((1+.95)/2)
low = mu-z*stan_error
high = mu+z*stan_error
#4
""" with a 95pr confidence interval, if we were to continue to take samples
we believe that 95percent of the time the observed mean will be between 2.1 and 2.26
"""
#5
""" If the sample size were smaller we would be less confident in using a normal distribution
our standard error would increase, increasing the size of our confidence interval.
With a sample size of 10, our normal assumption no longer stands. We would want to use a
t-distribution"""


#Part 2

#1

''' given a small sample size have large standard error and thus less precise CI, we want to
resample from the data to create a sampling distribution with more samples to reduce standard error
this giving us a more precise CI

'''


#2


def bootstrap(dat,resamp=10000):
    boot=[]
    for i in xrange(resamp):
        newdat=[dat[np.random.randint(0,len(dat))] for _ in xrange(len(dat))]
        boot.append(newdat)
    return boot

#PART 3

filepath = "/Users/twilightidol/Downloads/data2/productivity.txt"
prod = np.loadtxt(filepath, delimiter=',')



#2
'''
only reporting the mean difference is not enough information to reject the null hypothesis that
there is a difference in productivity as no statistical significance can be inferred, we also need
to know the sample size, variance etc.
'''

#3
mu_p = np.mean(prod)
stan_error_p = np.std(prod)/(len(prod)**.5)
 #95% confident interval is
z=scs.norm.ppf((1+.95)/2)
low_p = mu_p-z*stan_error_p
high_p = mu_p+z*stan_error_p
""" with a 95pr confidence interval, if we were to continue to take samples
we believe that 95percent of the time the observed mean will be between -.22 and 10.3
"""

#4
""" The sample size is not large enough. The standard error is too big"""

#5
def boostrap_ci(dat, funct,resamp=10000,cf=95):
    samp=bootstrap(dat,resamp=resamp)
    samp_stats=[np.mean(x) for x in samp ]
    low_p = np.percentile(samp_stats,(100-cf)/2)
    high_p = np.percentile(samp_stats,100-(100-cf)/2)
    return (funct(samp_stats),low_p,high_p)

boostrap_ci(prod, np.mean,resamp=10000)

#6

def bootstrap_mean(dat,resamp=10000):
    boot=[]
    for i in xrange(resamp):
        newdat=[ dat[np.random.randint(0,len(dat))] for i in xrange(len(dat))]
        boot.append(np.mean(newdat))
    return boot

# plt.hist(bootstrap_mean(prod))
# plt.show()

#7
#the bootstrap CI suggest the CI for the sampled mean
#amount of worth from changing the monitor is between 2000*4.785 and 2000*5.32
#and the cost of changing monitor is 100*500, there fore the cost outweights the
#benefit and does not recommend switching


#PART 4

filepath = "/Users/twilightidol/Downloads/data2/law_sample.txt"
law = np.loadtxt(filepath, delimiter=' ')

x=sp.array([d[0] for d in law])
y=sp.array([d[1] for d in law])

r_row, p_value = scs.pearsonr(x, y)

z=scs.norm.ppf((1+.95)/2)


se=(1/(len(x)))**0.5





def boostrap_ci_r(dat, funct,resamp=10000,cf=95):
    samp = bootstrap(dat, resamp=resamp)
    r_vals=[]
    for ls in samp:
        x=[d[0] for d in ls]
        y=[d[1] for d in ls]
        r_row, p_value = funct(x,y)
        r_vals.append(r_row)

    low_p = np.percentile(r_vals,(100-cf)/2)
    high_p = np.percentile(r_vals,100-(100-cf)/2)

    return (r_vals,low_p,high_p)



boot_r=boostrap_ci_r(law, scs.pearsonr,resamp=10000,cf=95)[0]

plt.hist(boot_r)

plt.show()



#5


filepath = "/Users/twilightidol/Downloads/data2/law_all.txt"
law_all = np.loadtxt(filepath, delimiter=' ')

x=sp.array([d[0] for d in law_all])
y=sp.array([d[1] for d in law_all])

r_row, p_value = scs.pearsonr(x, y)

#yes it is within the range of value for CI

