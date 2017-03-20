import numpy as np

A=np.array([[0.7,0.1,0],[0.2,0.9,0.2],[0.1,0,0.8]])

c=np.array([0.25,.2,0.55]).reshape(3,1)

prob=np.dot(A,c)
