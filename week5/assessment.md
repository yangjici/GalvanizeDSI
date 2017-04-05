1. Holding other coefficients constant, if the student is home-schooled, his SAT score will on average decrease by 40 points.
To justify our finding, we must check if the beta coefficient is statistically significant. Several factors may compromise
the finding because the regression assumption is not met, we must check for several
assumptions that linear regression hold in the data, such as normality of error, linearity, independence, etc. 

2. If the kid is home schooled, while holding other predictors constant, 
the odds of acceptance on average is multiplied by exp(-0.3) = 0.74, that is, we predict 
homeschooled kids will less likely be be accepted into a 4 year college than otherwise on average. 
and no we cannot say that 30 percent less home schooled kids are admitted
the probability is different from odds ratio, and we cannot make the prediction from a group level, but from individual level
since we must hold other variables fixed. We justify the finding by checking for statistical significance of the beta coefficient

3.

                  Predicted Negative  Predicted Positive

Actually Negative           30                10

Aactually positive          250              100

4.

                  Predicted Negative  Predicted Positive

Actually Positive            200              20

Aactually Negative          80000             200


5. we want to choose model B because we want to set our FP rate lower while still having a decent TP rate

6. We want to choose model A because we wnt to set our TP rate higher while still having a decent FP rate

7. splitting on number at 1

8.

9.The decision tree have high variance due to the tree overfit the model and chased too much noise in the data, we want 
to do some pruning, such as depth control, merging and cross validation to fix the high variance issue.

10. Decision trees in random forest uses a boostraped version of the training data, in addition each tree splits the data
based on randomly choosen part of the features instead of the entire feature set in each split. Decision tree uses the entire
dataset and at each split based the information gain on the entirety feature set.

11. We are able to CV our random forest because on average at each boostrap with replacement we get 2/3 of our dataset
we can then use the performance of the tree on the unchoosen 1/3 part of the data test that the tree have not seen to test
the performance of the tree.

12. Random forest only, the rest requires serial processing

13. we expect a max of 2*n leaf node are produced with max_depth set to n, if we set max_leaf_nodes to 2*n 
we would get about the same result

14. n_estimator dictates the number of trees used to serially predict on the error of previous tree, learning rate refers to 
the weight on the learning function of each tree.


15. Tuning max_features and sub_sample to having each tree focus on a part of the features or part of the samples
to make split decorrelate each tree
allowing each tree to have between tree variance in order to reduce the overall variance in prediction to prevent overfitting

16. In boosting we expect test error to decrease then increase as we build more trees  
In random forest, we expect test error to decrease in general as we build more trees

17. Lasso/Ridge: Lambda, as we increase lambda, we expect bias to increase and variance to decrease
SVM: C, as we decrease C, small margin, lower bias, higher variance

