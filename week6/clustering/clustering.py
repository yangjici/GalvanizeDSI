import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from pylab import ion
#1
df = pd.read_pickle('articles.pkl')

# mat=df.as_matrix()

# get the content column
content = df.content.as_matrix()

# vectorize our content
vector= TfidfVectorizer()

# vector=CountVectorizer()

out = vector.fit_transform(content)

# Feed input into our KMeans algo
cluster = KMeans(n_clusters=9)

cluster.fit(out)

df['label'] = cluster.labels_
df.groupby('label')['section_name'].value_counts()

#2

cluster.cluster_centers_

#3

important_words=[]

for cent in cluster.cluster_centers_:
    index =cent.argsort()[-10:][::-1]
    words = [vector.vocabulary_.keys()[i] for i in index]
    important_words.append(words)

#4

vector= TfidfVectorizer(max_features=10000, stop_words = 'english')

# vector=CountVectorizer()

out = vector.fit_transform(content)

# Feed input into our KMeans algo
cluster = KMeans(n_clusters=9)

cluster.fit(out)

df['label'] = cluster.labels_
df.groupby('label')['section_name'].value_counts()


important_words=[]

for cent in cluster.cluster_centers_:
    index =cent.argsort()[-10:][::-1]
    words = [vector.vocabulary_.keys()[i] for i in index]
    important_words.append(words)


#7.

out = vector.fit_transform(content)

# Feed input into our KMeans algo
cluster = KMeans(n_clusters=12)

cluster.fit(out)

df['label'] = cluster.labels_
df[df['source']=='The New York Times'].groupby('label')['section_name'].value_counts()


# Hierarchical clustering

# 1.
X_train, X_test= train_test_split(df, train_size = .15,random_state = 12)

X_train.groupby("section_name")['section_name'].value_counts()

#2

out_train = vector.fit_transform(X_train['content'])

out_train=out_train.todense()

Y = pdist(out_train)

Y= squareform(Y)

linkage = hierarchy.linkage(Y)

dendrogram(linkage,labels=list(X_train['section_name']))

#Hierarchical Topics

#1. need to add labels_

#2
