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

from scipy.interpolate import griddata
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

############################### LASSO and Random Forest ###############################

lasso = linear_model.Lasso(alpha=0.01)
lasso.fit(X_smote, y_smote)
y_pred = lasso.predict(X_OOS_test)
# turn the continous value into classification via simple >= 0.5 is 1
y_pred_classification = [1 if x >= 0.5 else 0 for x in y_pred]
print("LASSO score : " , )
print("LASSO Model Recall : " , recall_score(y_OOS_test, y_pred_classification))
print("LASSO Model Precision : ", precision_score(y_OOS_test,y_pred_classification))
print("--")

# See LASSO coefficient that is 0
lasso_coefficients = pd.Series(lasso.coef_)
non_zero_lasso_coefficients = lasso_coefficients[lasso_coefficients != 0]
# Print the non-zero coefficients
print("Non-zero Lasso Coefficients:")
print(non_zero_lasso_coefficients)

X_lasso_rf_train = X_smote[['loan_amount','property_value','income']]
X_lasso_rf_test = X_OOS_test[['loan_amount','property_value','income']]
random_forest = RandomForestClassifier(n_estimators = 100, max_depth=5, random_state=0)
random_forest.fit(X_lasso_rf_train, y_smote)
y_pred = random_forest.predict(X_lasso_rf_test)
print("RND Forest Model Recall : " , recall_score(y_OOS_test, y_pred))
print("RND Forest Precision : ", precision_score(y_OOS_test, y_pred))

############################### Cross Validation for LASSO and randomforest ###############################

# CV in Trees
# Set Hyperparameter (Lambda) values to cross validate here !!!!!
#max_depth = [10, 20, 30, 50]
#number_of_trees = [100, 150, 300, 500]
max_depth = [20]
number_of_trees = [1,10,20,30,40,50,60,70,80,90,100]

cross_validate_result = create_dictionary(number_of_trees, max_depth)
cross_validate_recall = create_dictionary(number_of_trees, max_depth)
cross_validate_precision = create_dictionary(number_of_trees, max_depth)
cross_validate_mse = create_dictionary(number_of_trees, max_depth)


for tree in number_of_trees:
    for depth in max_depth:
        print('Depth of Tree : ', depth, ' Number of Trees ', tree)
        accuracies = []
        recall_scores = []
        precision_scores = []
        mse_scores = []
        random_forest_cv = RandomForestClassifier(n_estimators=tree, max_depth=depth)
        for train_index, test_index in cv.split(X_smote):
            # change to loc to define the rows in the dataframe
            X_cv_train, X_cv_test, y_cv_train, y_cv_test = X_lasso_rf_train.iloc[train_index], X_lasso_rf_train.iloc[
                test_index], y_smote.iloc[train_index], y_smote.iloc[test_index]
            random_forest_cv.fit(X_cv_train, y_cv_train)
            y_pred = random_forest_cv.predict(X_cv_test)

            # Cross-Validation Prediction Error
            score = random_forest_cv.score(X_cv_test, y_cv_test)
            accuracies.append(score)
            recall_scores.append(recall_score(y_cv_test, y_pred))
            precision_scores.append(precision_score(y_cv_test, y_pred))
            mse_scores.append(mean_squared_error(y_cv_test, y_pred))

        cross_validate_result[tree][depth] = (sum(accuracies) / len(accuracies))
        cross_validate_recall[tree][depth] = (sum(recall_scores) / len(recall_scores))
        cross_validate_precision[tree][depth] = (sum(precision_scores) / len(precision_scores))
        cross_validate_mse[tree][depth] = (sum(mse_scores) / len(mse_scores))

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


############################### Plotting ###############################

# create separate lists for plotting: one for each metric (accuracy, recall, precision)
trees, depths, accuracies, recalls, precisions = [], [], [], [], []

# Extracting data from the dictionaries
for tree in number_of_trees:
    for depth in max_depth:
        trees.append(tree)
        depths.append(depth)
        accuracies.append(cross_validate_result[tree][depth])
        recalls.append(cross_validate_recall[tree][depth])
        precisions.append(cross_validate_precision[tree][depth])

print(trees)
print(depths)
print(accuracies)

############################### 3D - Point Scatter Plot ###############################

## Creating a 3D plot with lines connecting points to the Recall (z-axis) plane for clarity
#fig = plt.figure(figsize=(12, 9))
#ax = fig.add_subplot(111, projection='3d')
#
## Scatter plot
#scatter = ax.scatter(depths, trees, recalls, c=recalls, cmap='viridis', marker='o', depthshade=False)
#
## Adding lines to indicate the recall value more clearly
#for i in range(len(recalls)):
#    ax.plot([depths[i], depths[i]], [trees[i], trees[i]], [0, recalls[i]], 'gray', linestyle='--', linewidth=1)
#
#ax.set_xlabel('Max Depth')
#ax.set_ylabel('Number of Trees')
#ax.set_zlabel('Recall')
#ax.set_title('Recall Values by Max Depth and Number of Trees')
#
## Color bar to indicate recall values
#cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
#cbar.set_label('Recall Value')
#
#plt.show()
#
##########################

############################### 3D - Surface Plot for Recall ###############################


# Identifying the point with the highest recall value
max_recall = max(recalls)
max_recall_index = recalls.index(max_recall)
max_depth_at_max_recall = depths[max_recall_index]
number_of_trees_at_max_recall = trees[max_recall_index]

# Creating grid values for interpolation based on the previous setup
grid_x, grid_y = np.meshgrid(np.linspace(np.min(depths), np.max(depths), 100),
                             np.linspace(np.min(trees), np.max(trees), 100))

# Interpolating recall values over the grid
grid_z = griddata((depths, trees), recalls, (grid_x, grid_y), method='cubic')

# Creating a 3D plot with the interpolated surface
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.7)

# Scatter plot for the actual data points
scatter = ax.scatter(depths, trees, recalls, c='red', marker='o')

# Annotating the point with the highest recall
ax.text(max_depth_at_max_recall, number_of_trees_at_max_recall, max_recall,
        f'  Max Recall\n  Depth: {max_depth_at_max_recall}\n  Trees: {number_of_trees_at_max_recall}\n  Recall: {max_recall:.2f}',
        color='blue')

ax.set_xlabel('Max Depth')
ax.set_ylabel('Number of Trees')
ax.set_zlabel('Recall')
ax.set_title('Interpolated Recall Surface by Max Depth and Number of Trees')

# Color bar to indicate recall values
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
cbar.set_label('Recall Value')

plt.show()

############################### 3D - Surface Plot for Precision ###############################

# Identifying the point with the highest precision value
max_precision = max(precisions)
max_precision_index = precisions.index(max_precision)
max_depth_at_max_precision = depths[max_precision_index]
number_of_trees_at_max_precision = trees[max_precision_index]

# Interpolating precision values over the grid
grid_z_precision = griddata((depths, trees), precisions, (grid_x, grid_y), method='cubic')

# Creating a 3D plot with the interpolated surface for precision
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Surface plot for precision
surf_precision = ax.plot_surface(grid_x, grid_y, grid_z_precision, cmap='viridis', edgecolor='none', alpha=0.7)

# Scatter plot for the actual data points
scatter_precision = ax.scatter(depths, trees, precisions, c='red', marker='o')

# Annotating the point with the highest precision
ax.text(max_depth_at_max_precision, number_of_trees_at_max_precision, max_precision,
        f'  Max Precision\n  Depth: {max_depth_at_max_precision}\n  Trees: {number_of_trees_at_max_precision}\n  Precision: {max_precision:.2f}',
        color='blue')

ax.set_xlabel('Max Depth')
ax.set_ylabel('Number of Trees')
ax.set_zlabel('Precision')
ax.set_title('Interpolated Precision Surface by Max Depth and Number of Trees')

# Color bar to indicate precision values
cbar_precision = fig.colorbar(surf_precision, shrink=0.5, aspect=5)
cbar_precision.set_label('Precision Value')

plt.show()

############################### 3D - Combining Recall and Precision ###############################

# For combining the precision and recall surfaces in one plot, and highlighting their intersection,
# we will interpolate both sets of values (precision and recall) over the same grid and identify the closest points.

# Interpolating recall values over the same grid (reusing grid_x and grid_y)
grid_z_recall = griddata((depths, trees), recalls, (grid_x, grid_y), method='cubic')

# Creating a 3D plot to display both surfaces
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Surface plot for precision
surf_precision = ax.plot_surface(grid_x, grid_y, grid_z_precision, cmap='viridis', edgecolor='none', alpha=0.5, label='Precision')

# Surface plot for recall
surf_recall = ax.plot_surface(grid_x, grid_y, grid_z_recall, cmap='plasma', edgecolor='none', alpha=0.5, label='Recall')

# Annotating the points with the highest precision and recall
ax.text(max_depth_at_max_precision, number_of_trees_at_max_precision, max_precision,
        f'  Max Precision\n  Depth: {max_depth_at_max_precision}\n  Trees: {number_of_trees_at_max_precision}\n  Precision: {max_precision:.2f}',
        color='blue')

ax.text(max_depth_at_max_recall, number_of_trees_at_max_recall, max_recall,
        f'  Max Recall\n  Depth: {max_depth_at_max_recall}\n  Trees: {number_of_trees_at_max_recall}\n  Recall: {max_recall:.2f}',
        color='red')

ax.set_xlabel('Max Depth')
ax.set_ylabel('Number of Trees')
ax.set_zlabel('Metric Value')
ax.set_title('Interpolated Precision and Recall Surfaces')

# Color bars for precision and recall
cbar_precision = fig.colorbar(surf_precision, shrink=0.5, aspect=5, pad=0.1)
cbar_precision.set_label('Precision Value')
cbar_recall = fig.colorbar(surf_recall, shrink=0.5, aspect=5, pad=0.02)
cbar_recall.set_label('Recall Value')

plt.show()

# ############################### 2D Diagram - Separate ###############################
#
# # Creating a DataFrame for easier manipulation
# data = {
#     'Number of Trees': np.tile(number_of_trees, len(max_depth)),
#     'Max Depth': np.repeat(max_depth, len(number_of_trees)),
#     'Accuracy': accuracies,
#     'Recall': recalls,
#     'Precision': precisions
# }
#
# df_plot = pd.DataFrame(data)
#
# # Plotting
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
#
# # Recall plot
# for depth in max_depth:
#     subset = df_plot[df_plot['Max Depth'] == depth]
#     ax[0].plot(subset['Number of Trees'], subset['Recall'], '-o', label=f'Max Depth {depth}')
#
# ax[0].set_title('Recall vs. Number of Trees')
# ax[0].set_xlabel('Number of Trees')
# ax[0].set_ylabel('Recall')
# ax[0].legend(title='Max Depth')
# ax[0].grid(True)
#
# # Precision plot
# for depth in max_depth:
#     subset = df_plot[df_plot['Max Depth'] == depth]
#     ax[1].plot(subset['Number of Trees'], subset['Precision'], '-o', label=f'Max Depth {depth}')
#
# ax[1].set_title('Precision vs. Number of Trees')
# ax[1].set_xlabel('Number of Trees')
# ax[1].set_ylabel('Precision')
# ax[1].legend(title='Max Depth')
# ax[1].grid(True)
#
# plt.tight_layout()
# plt.show()
#
# ############################### 2D Diagram - Combination ###############################
#
# plt.figure(figsize=(12, 7))
#
# # Define marker styles to differentiate between Recall and Precision
# marker_styles = ['o', 's']
#
# # Plotting Recall and Precision for each Max Depth with automatic color assignment
# for i, depth in enumerate(max_depth):
#     subset = df_plot[df_plot['Max Depth'] == depth]
#     # Recall
#     plt.plot(subset['Number of Trees'], subset['Recall'], marker=marker_styles[0], linestyle='-', label=f'Recall - Depth {depth}')
#     # Precision
#     plt.plot(subset['Number of Trees'], subset['Precision'], marker=marker_styles[1], linestyle='--', label=f'Precision - Depth {depth}')
#
# plt.title('Recall and Precision vs. Number of Trees for Different Depths')
# plt.xlabel('Number of Trees')
# plt.ylabel('Metric Value')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Metric - Max Depth')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

################################ 2D Diagram - Separate ###############################

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 15))

# Recall
for tree in number_of_trees:
    recalls = [cross_validate_recall[tree][depth] for depth in max_depth]
    axs[0].plot(max_depth, recalls, label=f'{tree} Trees')
axs[0].set_title('Recall vs Max Depth')
axs[0].set_xlabel('Max Depth')
axs[0].set_ylabel('Recall')
axs[0].legend()

# Precision
for tree in number_of_trees:
    precisions = [cross_validate_precision[tree][depth] for depth in max_depth]
    axs[1].plot(max_depth, precisions, label=f'{tree} Trees')
axs[1].set_title('Precision vs Max Depth')
axs[1].set_xlabel('Max Depth')
axs[1].set_ylabel('Precision')
axs[1].legend()

# MSE
for tree in number_of_trees:
    mses = [cross_validate_mse[tree][depth] for depth in max_depth]
    axs[2].plot(max_depth, mses, label=f'{tree} Trees')
axs[2].set_title('MSE vs Max Depth')
axs[2].set_xlabel('Max Depth')
axs[2].set_ylabel('MSE')
axs[2].legend()

plt.tight_layout()
plt.show()

################################ 2D Diagram - Combine ###############################

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['blue', 'green', 'red']
metrics = ['Recall', 'Precision', 'MSE']
metric_dicts = [cross_validate_recall, cross_validate_precision, cross_validate_mse]

for i, metric_dict in enumerate(metric_dicts):
    for tree in number_of_trees:
        values = [metric_dict[tree][depth] for depth in max_depth]
        ax.plot(max_depth, values, label=f'{tree} Trees - {metrics[i]}', color=colors[i],
                linestyle='--' if tree == 5 else '-')

ax.set_title('Recall, Precision, and MSE vs Max Depth')
ax.set_xlabel('Max Depth')
ax.set_ylabel('Metric Values')
ax.legend()

plt.show()