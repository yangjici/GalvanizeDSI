import numpy as np
import logistic_regression_functions as f


class GradientDescent(object):
    """Preform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """

    def __init__(self, cost, gradient, predict_func,
                 alpha=0.01,
                 num_iterations=10000,fit_intercept=True):
        """Initialize the instance attributes of a GradientDescent object.

        Parameters
        ----------
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimizaiton has
            converged.
        alpha: The learning rate.
        num_iterations: Number of iterations to use in the descent.

        Returns
        -------
        self: The initialized GradientDescent object.
        """
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.intercept = fit_intercept

    def run(self, X, y):
        """Run the gradient descent algorithm for num_iterations repititions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        if self.intercept:
            X= f.add_intercept(X)
        self.coeffs= np.zeros(X.shape[1]).reshape(X.shape[1],1)

        for i in range(self.num_iterations):
            self.coeffs = self.coeffs - self.alpha* self.gradient(X,self.coeffs,y)
        return self

    def predict(self, X):
        """Call self.predict_func to return predictions.

        Parameters
        ----------
        X: Data to make predictions on.

        Returns
        -------
        preds: A one dimensional numpy array of predictions.
        """
        if self.intercept:
            X= f.add_intercept(X)
        return self.predict_func(X,self.coeffs)
