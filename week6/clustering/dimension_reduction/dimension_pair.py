import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd
from sklearn.decomposition import PCA
import numpy as np

#PART 1
#1
df = pd.read_csv('book_reviews.csv')
#2

user = df.pivot(index='User-ID',columns='ISBN',values='Book-Rating')

user = user.fillna(-1)

#PART 2

#1

U, sigma, VT = svd(user)
#2

power = sigma**2
#3

plt.plot(power[:500])
plt.yscale('log')
#plt.show
#4
cusum = np.cumsum(power)

plt.plot(cusum)

total = np.sum(power)

#5

ninety=0.9*total

np.sum(cusum < ninety)
#441
#6

U = U[:,:420]
sigma = sigma[:420]
VT = VT[:420,:]

VT2 = np.argsort(VT[::-1])
topten = VT2[:,:10]

#6

meta_data = pd.read_csv('book_meta.csv', sep=";", error_bad_lines=False)

for row in topten:
    isbn = user.columns.values[row]
    author_title=meta_data[meta_data['ISBN'].isin(isbn)][['Book-Title','Book-Author']]
    print author_title

#7


U_firstten = U[:5]
sigma_firstten = sigma[:5]
VT_firstten = VT[:5,:]

#8

user=user.reset_index()



U_firstten_indices = np.argsort(U_firstten[::-1])
topten_of_topfivetopics = U_firstten_indices[:,:10]
for row in topten_of_topfivetopics:
    isbn = user.columns.values[row]
    author_title=meta_data[meta_data['ISBN'].isin(isbn)][['Book-Title','Book-Author']]
    print author_title
