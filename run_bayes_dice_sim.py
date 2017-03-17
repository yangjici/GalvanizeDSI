
#create two instances of class object Bayes
sim_imb = Bayes(dice.prior_unbal.copy(),likelihood_func)
sim_uni = Bayes(dice.prior_uni.copy(),likelihood_func)

#call into method of update to simulate one or more if list dice roll(s):
sim_imb.update(8)
sim_uni.update(8)

dat = [2,3,5,6]
for d in dat:
  sim_imb.update(d)
  sim_uni.update(d)

sim_imb.print_distribution()
sim_uni.print_distribution()
