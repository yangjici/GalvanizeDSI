import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

with open('train.txt') as train:
    train_data = train.read()

with open('labels.txt') as labels:
    labels_data = labels.read()

all_descriptions = train_data.split('\n')

all_descriptions = all_descriptions[:len(all_descriptions)-1]


all_labels = labels_data.split('\n')

all_labels = all_labels[:len(all_labels)-1]


stop_words=stopwords.words('english')


all_content = []

for desc in all_descriptions:
    token = word_tokenize(''.join(desc))
    words=[x.lower() for x in token if (x.lower() not in stop_words and x.isalpha())]
    all_content.append(words)

snowball = SnowballStemmer('english')

#stemming

stem_list =[]

for doc in all_content:
    if doc!=[]:
        stem_list.append([snowball.stem(word) for word in doc])
    else:
        stem_list.append([])


stem_list = np.matrix(stem_list)


vectorizer = TfidfVectorizer()
