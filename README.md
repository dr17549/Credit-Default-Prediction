# Loan Default Prediction 
Data set from https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data

## Why is credit default prediction important?

Credit default prediction is a big part of lending and risk control. Financial instituions are very concerned with assesing risk of an invidual or company before issuing a loan. This is because non-performing loans are a huge financial distress for the company and collecting debts on these loans is a a burden in operational cost. 

## Why is it hard? 
The challenge with this problem is that in general, the number of defaulters are generally very smalll compared to the number of non-defaulters. The dataset is heavily imbalanced. 

In addition, because we are more concerned with correctly predicting those who are defaulters right, accuracy may not be the right metric to evalute the models. Since False Negative (those that are actually defaulters who were predicted as not default) is more important than those who are False Positive (those who are non-default who were predicted as defaulters), we rely on recall as the chosen performance measurement. 

## Process Overview 
We try and predict loan default using Machine Learning techniques. The steps we took were : 

1. Clean the dataset 
   1. Remove N/As
   2. Remove redundant variables
   3. Remove non-informative variables (IDs, Year)
   4. Remove outliers
5. Check for Correlation between independent variables 
2. Split Test Train dataset
3. SMOTE to oversample the data
4. Train Models and Cross Validate for hyperparameter tuning 

## Models 
1. OLS , LASSO and Ridge for Regularization
2. KNN
3. Random Forest
4. XGBoost 
