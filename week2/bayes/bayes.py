import matplotlib.pyplot as plt
import numpy as np
import dice

class Bayes(object):
    '''
    INPUT:
        prior (dict): key is the value (e.g. 4-sided die),
                      value is the probability
        likelihood_func (function):
                        INPUT: INT data--the value of the roll of a die
                               INT key--the maximum value of the die (ie pass
                                            4 to get the likelihood for a
                                            4-sided die)
    '''
    def __init__(self, prior, likelihood_func):
        self.prior = prior
        self.likelihood_func = likelihood_func


    def normalize(self):
        s=sum(self.prior.values())
        for k,v in self.prior.iteritems():
            self.prior[k]=v/s
        '''
        INPUT: None
        OUTPUT: None
        Makes the sum of the probabilities equal 1.
        '''


    def update(self, data):
        for k,v in self.prior.iteritems():
            self.prior[k] *= self.likelihood_func(data,k)
        self.normalize()
        '''
        INPUT:
            data (int or str): A single observation (data point)
        OUTPUT: None
        Conduct a bayesian update. Multiply the prior by the likelihood and
        make this the new prior.
        '''


    def print_distribution(self):
        for k,v in self.prior.iteritems():
            print "dice: {} probability: {}" .format(k,v)
        '''
        Print the current posterior probability.
        '''



    def plot(self, color=None, title=None, label=None):
        '''
        Plot the current prior.
        '''
        pass
