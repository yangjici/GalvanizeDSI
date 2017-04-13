import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.decomposition

import matplotlib.pyplot as plt
import nmf


df=pd.read_pickle('articles.pkl')

content = df.content.as_matrix()

content = df.content.as_matrix()

n_topics = 7

# vectorize our content
vector = TfidfVectorizer(max_features=5000, stop_words='english')

vector_matrix = vector.fit_transform(content).toarray()

features = np.array(vector.get_feature_names())

our_nmf = nmf.NMF(vector_matrix, n_topics, max_iter=100)

W, H = our_nmf.fit()

### Using Your NMF Function
print "MSR: {}".format(our_nmf.msr())

for row in H:
    ind=row.argsort()[:-10-1:-1]
    print features[ind]


### Built-In NMF
print "\nsklearn.NMF\n"

skl_nmf = sklearn.decomposition.NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5, max_iter=100)
skl_W = skl_nmf.fit_transform(vector_matrix)
skl_H = skl_nmf.components_

print "MSR: {}".format(skl_nmf.reconstruction_err_)

for row in skl_H:
    ind=row.argsort()[:-10-1:-1]
    print features[ind]
