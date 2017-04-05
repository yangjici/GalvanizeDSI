from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np


client = MongoClient()
db = client.nyt_dump
coll = db.articles
all_articles=[]

for document in coll.find():
    all_articles.append(document['content'])

stop_words=stopwords.words('english')

all_content = []

for article in all_articles:
    token = word_tokenize(''.join(article))
    words=[x.lower() for x in token if (x.lower() not in stop_words and x.isalpha())]
    all_content.append(words)

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

# def stem(chopper, list_of_words):
#
#     stem_list =[]
#     for doc in all_content:
#         stem_list.append([chopper.stem(word) for word in doc])
#     return stem_list
def stem_and_lem(chopper, list_of_words, stem=True):

    stem_list =[]
    if stem:
        for doc in all_content:
            stem_list.append([chopper.stem(word) for word in doc])
    else:
        for doc in all_content:
            stem_list.append([chopper.lemmatize(word) for word in doc])
    return stem_list

porter_stem = stem_and_lem(porter, all_content)
snowball_stem = stem_and_lem(snowball, all_content)
wordnet_lem = stem_and_lem(wordnet, all_content, stem=False)



bag_of_words = list(set([word for doc in porter_stem for word in doc]))

rev_lookup = {word:idx for idx, word in enumerate(bag_of_words)}

count_matrix = np.zeros((len(all_articles),len(bag_of_words)))

'''
Now let's create our word count vectors manually.
Create a numpy matrix where each row corresponds to a document
and each column a word. The value should be the count of the
number of times that word appeared in that document.

'''

for i in range(len(all_articles)):
    count=Counter(porter_stem[i])
    for k in count:
        j=rev_lookup[k]
        count_matrix[i][j]+=count[k]


#for each article count the word
#for each word, look up index
#in every for loop keep in mind the number of row

#apply laplace smoothing constant of 1
doc_freq=Counter()

for article in porter_stem:
    count = Counter(article)
    for k in count:
        doc_freq[k]+=1

#normalize the matrix

norm=np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)),1, count_matrix).reshape(len(all_articles),1)

norm_matrix=np.divide(count_matrix,norm)
