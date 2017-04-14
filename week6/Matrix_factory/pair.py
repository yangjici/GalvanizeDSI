import graphlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/u.data', sep='\t', header=None)
df = df.drop(3,axis=1)

sframe = graphlab.SFrame(df)

fact_rec = graphlab.recommender.factorization_recommender.create(observation_data=sframe, user_id=0, item_id=1, target=2,solver='als')

one_datapoint_sf = graphlab.SFrame({'0' :[1], '1': [100]})
fact_rec.predict(one_datapoint_sf)
'''
Out[31]:
dtype: float
Rows: 1
[4.9138281148529055]
'''

fact_rec.get('coefficients')['0']

intercept = fact_rec.get('coefficients')['intercept']


res = fact_rec.get('coefficients')['0']['0']
res  = fact_rec.get('coefficients')['0'][res==1]

u_factors = np.array(res['factors'])

res = fact_rec.get('coefficients')['1']['1']
res  = fact_rec.get('coefficients')['1'][res==100]

v_factors = np.array(res['factors'])

intercept+np.dot(u_factors,v_factors.T)

#same as result

predicted_res = np.array(fact_rec.predict(sframe))

actual_ratings = np.array(sframe['2'])

rmse = (np.sum((predicted_res-actual_ratings)**2)/len(predicted_res))**0.5


pd.Series(predicted_res).describe()

pd.Series(actual_ratings).describe()


# we have negative value in our predicted rating, in addition, the max
#value in our predicted rating exceeds the rating range

fig, axe = plt.subplots()

for ax,dat in zip(axes,[actual_ratings,predicted_res]):
    ax.violinplot(dat)

#regularization

fact_rec1 = graphlab.recommender.factorization_recommender.create(observation_data=sframe, user_id=0, item_id=1, target=2,solver='als',regularization= 0.001,random_seed=666)

'''
Q:Notice, that graphlab provides two regularization parameters. The parameter regularization controls the value of lambda and is essentially changing the target function from
A:
The lambda value acts a penalization on our factors, on the size of the U-> and V-> at each iteration
to minimize the cost function. That means we're going to see a higher RMSE, because it hasn't fit
as precisely to the input data. That also, means our predictions are more "cautious"



'''
