'''
INPUT:
    prior (dict): key is the value (e.g. 4-sided die),
                  value is the probability

    likelihood_func (function):
                    INPUT: INT data--the value of the roll of a die
                           INT key--the maximum value of the die (ie pass
                                        4 to get the likelihood for a
                                        4-sided die)
    likelihood(data, key):
'''
prior_unbal={"4":0.08,"6":0.12,"8":0.16,"12":0.24,"20":0.4}

prior_uni={"4":0.2,"6":0.2,"8":0.2,"12":0.2,"20":0.2}

def likelihood_func(roll,die):
    if roll > int(die):
        return 0
    else:
        return 1/(float(die))
