import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''
        sample_weight = np.ones(x.shape[0])/float(x.shape[0])
        for m in xrange(self.n_estimator):
            estimator,sample_weight,alpha = self._boost(x,y,sample_weight)
            self.estimators_.append(estimator)
            self.estimator_weight_[m]=alpha





          ### YOUR CODE HERE ###


    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.
        '''
        #m = tree number  (m of M)
        #i = data point number (i of N)


        #a
        estimator = clone(self.base_estimator)
        estimator.fit(x, y, sample_weight=sample_weight)

        #b
        numerator = np.sum( sample_weight*(y != estimator.predict(x)) )
        denominator = np.sum(sample_weight)
        error = numerator / float(denominator)

        #c
        alpha = np.log((1-error) / error)

        #d
        sample_weight *= np.exp(alpha * (y != estimator.predict(x)))

        return estimator, sample_weight,alpha


    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''

        predictions = []
        actual_prediction = []
        for tree, alpha  in zip(self.estimators_,self.estimator_weight_):
            tree_pred= alpha*(tree.predict(x)*2-1)
            predictions.append(tree_pred)

        for pred in np.array(predictions).T:
            tot=np.sum(pred)
            tot = 0 if tot <0 else 1
            actual_prediction.append(tot)

        # for row in x:
        #     pred = np.sum([alpha* (tree.predict(row)*2-1) for alpha, tree in zip(self.estimator_weight_, self.estimators_)])
        #     pred = 0 if pred < 0 else 1
        #     predictions.append(pred)

        return np.array(actual_prediction)


    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        pred= self.predict(x)
        score = np.sum(pred == y)/float(len(y))
        return score
