from bandits import Bandits
from banditstrategy import BanditStrategy
import matplotlib.pyplot as plt
import numpy as np


bandits = Bandits([0.05, 0.03, 0.06])
strategies = ['epsilon_greedy', 'softmax', 'ucb1', 'bayesian_bandit']
strat = BanditStrategy(bandits, "bayesian_bandit")
strat.sample_bandits(1000)
print("Number of trials: ", strat.trials)
print("Number of wins: ", strat.wins)
print("Conversion rates: ", strat.wins / strat.trials)
print("A total of %d wins of %d trials." % (strat.wins.sum(), strat.trials.sum()))

# bands = [[0.1, 0.1, 0.1, 0.1, 0.9],
#         [0.1, 0.1, 0.1, 0.1, 0.12],
#         [0.1, 0.2, 0.3, 0.4, 0.5]]
#
# for b in bands:
#     for s in strategies:
#         bandits = Bandits(b)
#         strat = BanditStrategy(bandits, s)
#         strat.sample_bandits(1000)
#         # print("Number of trials: ", strat.trials)
#         # print("Number of wins: ", strat.wins)
#         print "Strategy", s, "Bandit", b
#         print("Conversion rates: ", strat.wins / strat.trials)
#         print("A total of %d wins of %d trials." % (strat.wins.sum(), strat.trials.sum()))


def regret(probabilities, choices):
    '''
    INPUT: array of floats (0 to 1), array of ints
    OUTPUT: array of floats

    Take an array of the true probabilities for each machine and an
    array of the indices of the machine played at each round.
    Return an array giving the total regret after each round.
    '''
    p_opt = np.max(probabilities)
    return np.cumsum(p_opt - probabilities[choices])

# c = map(int, strat.choices)
# regrets=regret(np.array([0.05, 0.03, 0.06]), np.array(strat.choices).astype(int))
# plt.plot(regrets)
# plt.show()

#4 percentage of time the optimal bandit was choosen


def percent(probabilities,choices):
    choices=map(int,choices)
    max_arg=np.argmax(probabilities)
    correct=np.array([1.0 if i==max_arg else 0 for i in choices])
    trials = np.array(range(1,len(choices)+1))
    return np.cumsum(correct)/np.cumsum(trials)

percentages=percent([0.05,0.03,0.06],strat.choices)

plt.plot(percentages)
plt.show()
