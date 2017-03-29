from collections import Counter
import numpy as np

array = [1, 1, 2, 1, 2]

array=np.array(array)


prop=np.array(Counter(array).values())/float(len(array))


1-np.sum(prop**2)


prop=np.array(Counter(array).values())/float(len(array))

-1* np.sum([a*np.log2(a) for a in prop])
