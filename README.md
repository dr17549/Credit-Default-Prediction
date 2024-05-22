# Loan Default Prediction 
Data set from https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data

## Why is credit default prediction important?


This dataset is Loan Default. The aim is to tune a model to predict whether a loan will default or not. The challenge with this problem
is that the dataset is highly imbalanced. The traditional method will not work very well and we cannot rely on accuracy as the only metric. In addition, since False Negative (those that are actually defaulters who were predicted as not default) is more important than those who are False Positive (those who are non-default who were predicted as defaulters). 


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
