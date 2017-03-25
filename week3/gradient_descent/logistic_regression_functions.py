import numpy as np
from sklearn.linear_model import LogisticRegression


def predict_proba(X, coeffs):
    """Calculate the predicted conditional probabilities (floats between 0 and
    1) for the given data with the given coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.

    Returns
    -------
    predicted_probabilities: The conditional probabilities from the logistic
        hypothosis function given the data and `coefficients.

    """
    X.dot(coeffs)

    predicted_probabilities = 1/(1+np.exp(-X.dot(coeffs)))
    return predicted_probabilities


def predict(X, coeffs, threas=0.5):
    """
    Calculate the predicted class values (0 or 1) for the given data with the
    given coefficients by comparing the predicted probabilities to a given
    threashold.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.
    threas: Threashold for comparison of probabilities.

    Returns
    -------
    predicted_class: The predicted class.
    """
    predicted_probabilities = predict_proba(X, coeffs)
    predicted = np.array([1 if a > 0.5 else 0 for a in predicted_probabilities])
    return predicted


def cost_log(h,y):
     return y*np.log(h)+(1-y)*np.log(1-h)

def cost(X, y, coeffs):
    """
    Calculate the logistic cost function of the data with the given
    coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    y: A 1 dimensional numpy array.  The actual class values of the response.
        Must be encoded as 0's and 1's.  Also, must align properly with X and
        coeffs.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X, y, and coeffs must align.

    Returns
    -------
    logistic_cost: The computed logistic cost.
    """
    Hx=predict_proba(X,coeffs)
    logistic_cost=-(sum([cost_log(h,y) for h,y in zip(Hx,y)]))
    return logistic_cost



def gradient(X,coeffs,y):
    Hx=predict_proba(X,coeffs)
    dif=Hx-y.reshape(len(y),1)
    return X.T.dot(dif)

def add_intercept(X):
    row=X.shape[0]
    return np.hstack((np.ones(row).reshape(row,1),X))
