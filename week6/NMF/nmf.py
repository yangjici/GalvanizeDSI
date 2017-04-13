from numpy.linalg import lstsq
import numpy as np


class NMF(object):


    def __init__(self,X,k,max_iter):
        self.X = X
        self.k = k
        self.max_iter = max_iter
        (n,m) = self.X.shape
        self.W = np.random.randint(100,size=(n,k))
        self.H = np.random.randint(100,size=(k,m))

    def fit(self):
        for i in range(self.max_iter):
            # import pdb; pdb.set_trace()

            self.H = lstsq(self.W,self.X)[0]
            self.H[self.H < 0] = 0

            self.W = (lstsq(self.H.T,self.X.T))[0].T
            self.W[self.W < 0] = 0

            if abs(np.sum(self.X - np.dot(self.W,self.H)))<0.001:
                break

        return self.W, self.H

    def msr(self):
        return np.sum((self.X - np.dot(self.W,self.H))**2).mean()

"""

Error =   min || b - a x ||^2

Error =   min || X - W*H ||^2
Error =  W.T  = MIN(  || X - H.T*W.T ||^2  )
"""
