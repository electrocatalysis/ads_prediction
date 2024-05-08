#!/usr/bin/env python
import pandas as pd
import numpy as np 
import time
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score     
from sklearn.model_selection import train_test_split 
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor 
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import RandomizedSearchCV
import warnings
import os 
import string 
import torch
warnings.filterwarnings("ignore")


# Read data
df = pd.read_excel('./Feature_CO.xlsx', sheet_name= "ML_features")
features = df.iloc[:,:-1]
target_CO = df.iloc[:,-1]

# Hyperparameter tuning

# Instantiate regressor algorithms
GBR = GradientBoostingRegressor(random_state=42)
model = GBR

# Feature normalization (standardize the descriptor)
features = (features - features.mean(axis=0)) / features.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(features, target_CO, train_size= 0.8, random_state=45)       

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

model_grid = {'loss':['ls', 'lad', 'huber', 'quantile'],
             'n_estimators': [10,50,100],
             'learning_rate':[0.05, 0.1, 0.15],
             'max_depth':[2,3,4],
             'min_samples_split':[2,3,5],
             'min_samples_leaf':[1,2,4]}
model.get_params().keys()

model_grid_cv = RandomizedSearchCV(estimator=model,
                        param_distributions=model_grid,
                        n_iter=100,
                        cv=10,
                        verbose= True)
model_grid_cv.fit(X_train,y_train)
model_optimized = model_grid_cv.best_estimator_
# print (model_optimized)


# Feature importance 
headers = features.columns.values.tolist()
importances = model_optimized.feature_importances_
sorted_idx = np.argsort(importances)
importances_2 = importances[sorted_idx]
headers_2 = np.array(headers)[sorted_idx]


fig, ax = plt.subplots( figsize=(15, 8))
font={'weight':'normal', 
      'size': 28}
plt.rc('font', **font)
plt.bar(headers_2, importances_2,)

# ax.set_xlabel("Feature Importance", fontsize = 28)
ax.set_ylabel("Feature Importance", fontsize = 28)
ax.tick_params(axis='y', )
ax.tick_params(axis='x', rotation = 90)

plt.tick_params(labelsize=24)

ax.spines['bottom'].set_linewidth(2);
ax.spines['left'].set_linewidth(2);  
ax.spines['right'].set_linewidth(2); 
ax.spines['top'].set_linewidth(2); 
plt.title('GBR feature importance')
plt.grid(axis='y', ls='--', alpha=0.5)
ax.set_ylim(0, 0.25)

plt.tight_layout()
fig.savefig('Feature importance.jpeg', dpi=600,)


# 500 trials repeat 

# Initialize lists to store metrics for training and testing
R2_2nd = []
RMSE_2nd = []
R2_2nd_test = []
RMSE_2nd_test = []

# Run the experiment 500 times
print('--- start 500 trials repeat ---')
for i in range(500):
    # Randomly split data
    X_train, X_test, y_train, y_test = train_test_split(features, target_CO, train_size=0.8)
    
    # Predictions and evaluation for training set
    y_tr_pred = model_optimized.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_tr_pred, y_train))
    R_squr_train = r2_score(y_train, y_tr_pred)
    
    # Store metrics for training set
    RMSE_2nd.append(rmse_train)
    R2_2nd.append(R_squr_train)
    
    # Predictions and evaluation for testing set
    y_te_pred = model_optimized.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_te_pred, y_test))
    R_squr_test = r2_score(y_test, y_te_pred)
    
    # Store metrics for testing set
    RMSE_2nd_test.append(rmse_test)
    R2_2nd_test.append(R_squr_test)

# Create DataFrames from the collected metrics
text1 = pd.DataFrame({'gbr-rmse': RMSE_2nd})
text2 = pd.DataFrame({'gbr-r2': R2_2nd})
text3 = pd.DataFrame({'gbr-rmse_test': RMSE_2nd_test})
text4 = pd.DataFrame({'gbr-r2_test': R2_2nd_test})

# Calculate and print average values for training set
avg_rmse_train = np.mean(RMSE_2nd)
avg_r2_train = np.mean(R2_2nd)

print('Average RMSE for training set: {:.3f} eV'.format(avg_rmse_train))
print('Average R^2 for training set: {:.3f}'.format(avg_r2_train))

# Calculate and print average values for testing set
avg_rmse_test = np.mean(RMSE_2nd_test)
avg_r2_test = np.mean(R2_2nd_test)

print('Average RMSE for testing set: {:.3f} eV'.format(avg_rmse_test))
print('Average R^2 for testing set: {:.3f}'.format(avg_r2_test))

# Concatenate DataFrames along columns
result_df = pd.concat([text1, text2, text3, text4], axis=1)

# Write the concatenated DataFrame to an Excel file
result_df.to_excel('GBR_500_trials_repeat_22D.xlsx', index=False)  