# import libraries

# handle dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


import pprint
from tabulate import tabulate

from sklearn import linear_model

from imblearn.over_sampling import SMOTE

from sklearn.metrics import recall_score, precision_score, f1_score, mean_squared_error


def create_dictionary(param_1,param_2):
    result_dictionary = {}
    for i in param_1:
        result_dictionary[i] = {}
        for j in param_2:
                result_dictionary[i][j] = {}
    return result_dictionary

############################### Read Dataset ###############################

df = pd.read_csv('Loan_Default.csv')
# remove some variables that are redundant
# year, ID
# Interest_rate_spread and rate_of_interest are dropped due to the fact that their values are NA where the loan has defaulted
# open credit , secured by, total units : only has one value or too few observations
df = df.drop(['year', 'ID', 'loan_type', 'loan_purpose', 'Interest_rate_spread',
              'rate_of_interest', 'open_credit','Upfront_charges','construction_type',
              'Secured_by', 'Security_Type','total_units'], axis=1)
# print(df['Status'].value_counts())
# print(df.isna().sum())

# Assumption 2 : We focus on mortgage loans
df = df.dropna(subset=["property_value"])
# Assumption 3 : drop approv in adv that is NA (only small set of samples)
df = df.dropna(subset=["term"])
df = df.dropna(subset=["loan_limit"])
df = df.dropna(subset=["approv_in_adv"])
df = df.dropna(subset=["Neg_ammortization"])
df = df.dropna(subset=["income"])
# Drop rows where sex is not available
df = df.drop(df[df['Gender'] == 'Not Available'].index)


categorical_columns = ['loan_limit', 'Gender', 'approv_in_adv', 'Credit_Worthiness',
       'business_or_commercial', 'Neg_ammortization',
       'interest_only', 'lump_sum_payment', 'occupancy_type',
        'credit_type', 'co-applicant_credit_type',
       'age', 'submission_of_application', 'Region']

for i in categorical_columns:
    df = pd.concat([df,pd.get_dummies(df[i],drop_first=True, prefix=i)],axis=1)
    df = df.drop(i,axis=1)

############################### Train & Test Split ###############################

y = df['Status']
X = df.drop('Status',axis=1)
X_train, X_OOS_test, y_train, y_OOS_test = train_test_split(X, y, test_size=0.20, random_state=66)

############################### Oversampling ###############################

# Over sample using SMOTE
# -- by inspecting the data, we see that the minority class is extremely class (fraud "Class" == 1)
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X_train, y_train)

############################### K-Fold Cross Validation Setup ###############################

n_splits = 5
shuffle = True
random_state = 809
cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
# plot = plot_cv_indices(cv, X_smote, y_smote, n_splits)

############################### KNN CV ###############################

# CV in Trees
# Set Hyperparameter (Lambda) values to cross validate here !!!!!
number_of_neighbours = list(range(1, 31))

cross_validate_result = {}
cross_validate_recall = {}
cross_validate_precision = {}
cross_validate_mse = {}

for neighbour in number_of_neighbours:
    print('Number of Neighbours : ', neighbour)
    accuracies = []
    recall_scores = []
    precision_scores = []
    mse_scores = []
    knn = KNeighborsClassifier(n_neighbors=neighbour)
    for train_index, test_index in cv.split(X_smote):
        # change to loc to define the rows in the dataframe
        X_cv_train, X_cv_test, y_cv_train, y_cv_test = X_smote.iloc[train_index], X_smote.iloc[test_index], \
        y_smote.iloc[train_index], y_smote.iloc[test_index]
        knn.fit(X_cv_train, y_cv_train)
        y_pred = knn.predict(X_cv_test)

        # Cross-Validation Prediction Error
        score = knn.score(X_cv_test, y_cv_test)
        accuracies.append(score)
        recall_scores.append(recall_score(y_cv_test, y_pred))
        precision_scores.append(precision_score(y_cv_test, y_pred))
        mse_scores.append(mean_squared_error(y_cv_test, y_pred))

    cross_validate_result[neighbour] = (sum(accuracies) / len(accuracies))
    cross_validate_recall[neighbour] = (sum(recall_scores) / len(recall_scores))
    cross_validate_precision[neighbour] = (sum(precision_scores) / len(precision_scores))
    cross_validate_mse[neighbour] = (sum(mse_scores) / len(mse_scores))

    print("Accuracy : " + str((sum(accuracies) / len(accuracies))))
    print("Precision : " + str((sum(recall_scores) / len(recall_scores))))
    print("Recall : " + str((sum(precision_scores) / len(precision_scores))))
    print("MSE : " + str((sum(mse_scores) / len(mse_scores))))
    print()
# At the end you'll see what the value of each LASSO paramter is
print('------------------')
print('Accuracy : ', cross_validate_result)
print('Precision : ', cross_validate_precision)
print('Recall : ', cross_validate_recall)
print('MSE : ', cross_validate_mse)


############################### Plotting - Separate ###############################

# Convert the dictionaries to lists for plotting
neighbors = list(cross_validate_precision.keys())
precision_scores = list(cross_validate_precision.values())
recall_scores = list(cross_validate_recall.values())
mse_scores = list(cross_validate_mse.values())

# Plotting the relationships
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Precision plot
axes[0].plot(neighbors, precision_scores, marker='o', linestyle='-', color='b')
axes[0].set_title('Number of Neighbors vs Precision')
axes[0].set_xlabel('Number of Neighbors')
axes[0].set_ylabel('Precision')
axes[0].grid(True)

# Recall plot
axes[1].plot(neighbors, recall_scores, marker='o', linestyle='-', color='r')
axes[1].set_title('Number of Neighbors vs Recall')
axes[1].set_xlabel('Number of Neighbors')
axes[1].set_ylabel('Recall')
axes[1].grid(True)

# MSE plot
axes[2].plot(neighbors, mse_scores, marker='o', linestyle='-', color='g')
axes[2].set_title('Number of Neighbors vs MSE')
axes[2].set_xlabel('Number of Neighbors')
axes[2].set_ylabel('MSE')
axes[2].grid(True)

plt.tight_layout()
plt.show()

############################### Plotting - Combine ###############################

# Combining the three plots into one, with a secondary axis for MSE due to scale difference

fig, ax1 = plt.subplots(figsize=(10, 6))

# Precision and Recall
ax1.plot(neighbors, precision_scores, marker='o', linestyle='-', color='b', label='Precision')
ax1.plot(neighbors, recall_scores, marker='x', linestyle='--', color='r', label='Recall')
ax1.set_xlabel('Number of Neighbors')
ax1.set_ylabel('Precision / Recall', color='k')
ax1.tick_params(axis='y')
ax1.grid(True)

# MSE on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(neighbors, mse_scores, marker='s', linestyle='-.', color='g', label='MSE')
ax2.set_ylabel('MSE', color='k')
ax2.tick_params(axis='y')

# Combined legend for both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

ax1.set_title('Number of Neighbors vs Precision, Recall, and MSE')
plt.show()