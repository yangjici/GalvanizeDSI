assessment 5

#question 1

def number_of_jobs(query):
    '''
    INPUT: string
    OUTPUT: int

    Return the number of jobs on the indeed.com for the search query.
    '''

    url = "http://www.indeed.com/jobs?q=%s" % query.replace(' ', '+')
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    num = soup.find(class_='resultsTop').div.text.split()[5]
    return int(num.replace(',',''))

#question 2

          actual True    False
predicted

True             -10       -10

False            -100       0

#question 3


model 1: -1500 + -5000 + -1500 = -8000

model 2: -2000 + -5000 + = -7000

model 2 would save us the most

#question 4

for doc1:
{dog: 2, like: 1, cat:1}
for all the words:
{dog:3, like:1, cat:2, chase:1, bicycle:3, ride:1, basket:1, fast:1}

#question 5

from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["Dogs like dogs more than cats.",
             "The dog chased the bicycle.",
             "The cat rode in the bicycle basket.",
             "I have a fast bicycle."]

vector= TfidfVectorizer(stop_words = 'english')

res = vector.fit_transform(documents)


res.toarray()

res.toarray().sum(axis=1)

Document 1 is most similar to document 2

#question 6

vect = TfidfVectorizer(stop_words='english')
tokenized =[]

for doc in documents:
  tokenized.append(nltk.tokenize(doc))

X = vect.fit_transform(tokenized)
X_train, X_test, y_train, y_test = train_test_split(X, y)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print "Accuracy on test set:", log_reg.score(X_test, y_test)

#question 7

we need to do laplace smoothing because certain words that exist in one document does not exist in another document, and since we are multiplying all the probabilities of term frequency together, if one probabilities is zero, the entire p(x|c) become zero, thus we need to apply laplace smoothing to prevent that from occuring

#question 8

D = 100**(1/3)

D = 4.64158

4.64185 = N**(1/10)

ln(4.64185) = ln(N)*(1/10)

e**15.3 = N

N = 4412711

#question 9

we can try multiple initializations and pick the one that has
lowest distance between the point to each centroid

or we can try kmeans++ that initialize centroid with decreasing probabilities of assigning next centroid as the distance from the other centroid increases

#10

in clustering, k is chosen before running algorithm and we pick the number of k up to the point after which we would get a drastic decrease in margin of return in within cluster similarity (elbow method)

in hierarchical clustering, we can choose the number of k after running the algorithm and merge the rest of the clusters between the hiearchy that we have chosen, we can pick the number of k also using the elbow method
