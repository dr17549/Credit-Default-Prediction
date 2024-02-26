# Loan Default prediction 
Data set from https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data

This dataset is Loan Default. The aim is to tune a model to predict whether a loan will default or not. The challenge with this problem
is that the dataset is highly imbalanced. The traditional method will not work very well and we cannot rely on accuracy as the only metric. In addition, since False Negative (those that are actually defaulters who were predicted as not default) is more important than those who are False Positive (those who are non-default who were predicted as defaulters). 

We try and predict loan default using Machine Learning techniques. The steps we took were : 

1. Clean the dataset 
   1. Remove N/As
   2. Remove redundant variables
   3. Remove non-informative variables (IDs, Year)
2. Split Test Train dataset
3. SMOTE to oversample the data
4. OLS , LASSO and Ridge for Regularization 
5. KNN 
6. Random Forest
7. Random Forest and LASSO 
8. Adaboost
9. Gradient Boosting
10. XGBoost
11. XGBoost with adjusting threshold 

In these models, we also cross validated the hyperparameters (such as lambda in LASSO and number of trees in Random Forest). 
