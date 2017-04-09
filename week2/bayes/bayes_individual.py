import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

with open("/Users/twilightidol/Downloads/data5/siteA.txt") as f:
    clickA = [int(a.strip()) for a in f]

with open("/Users/twilightidol/Downloads/data5/siteB.txt") as f:
    clickB = [int(a.strip()) for a in f]

x = np.arange(0, 1.01, 0.01)

y_uni=scs.beta.pdf(x,a=1,b=1)

y_beta = scs.beta.pdf(x,a=6,b=46)


plotA=clickA[:50]

def plot_with_fill(x, y, label):
    lines = plt.plot(x, y, label=label, lw=2)
    plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())
    plt.legend()

#4 making multiple plots


# plot_with_fill(x, y_uni, label="uniform prior")
#
#
# index=[50,100,200,400,800]
#
# ls=[]
#
# for ind in index:
#     ls.append(clickA[:ind])


# for obs in ls:
#     n=len(obs)
#     k=sum(obs)
#     x = np.arange(0, 1.01, 0.01)
#     y_beta = scs.beta.pdf(x,a=k+1,b=n-k+1)
#     plot_with_fill(x,y_beta,label="{} observations".format(n))



#7
# for obs in [clickA,clickB]:
#     n=len(obs)
#     k=sum(obs)
#     x = np.arange(0, 1.01, 0.001)
#     y_beta = scs.beta.pdf(x,a=k+1,b=n-k+1)
#     plot_with_fill(x,y_beta,label="{} observations".format(n))


na=len(clickA)
ka=sum(clickA)

nb=len(clickB)
kb=sum(clickB)

A=np.random.beta(ka+1,na-ka+1,size=10000)

B=np.random.beta(kb+1,nb-kb+1,size=10000)

dif=B-A

prop=sum([1 for d in dif if d >0 ] ) /float(10000)


#9

scs.beta.ppf([0.025,0.975],ka+1,na-ka)


#10

dif2=(B-A)

prop2=sum([1 for d in dif2 if d >0.02 ] ) /float(10000)


plt.hist(dif)
