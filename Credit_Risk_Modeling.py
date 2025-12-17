# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:41:40 2025

@author: Mohd Uzaif
"""

# Import necessary libraries.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer



# Load the data from excel file.
internal_data = pd.read_excel('./Datasets/case_study1.xlsx')
cibil_data = pd.read_excel('./Datasets/case_study2.xlsx')


# Make a copy of our data so that I can change in it easily.
internal_df = internal_data.copy()
cibil_df = cibil_data.copy()


# check the shape of both the dataframes.
internal_df.shape
cibil_df.shape


# remove all the NaN(-99999) values from the internal_df.
internal_df = internal_df[internal_df['Age_Oldest_TL'] != -99999]

# columns removed that have more than 10000 values are NULL or -99999.
columns_to_be_removed = []
for column in cibil_df.columns:
    if (cibil_df[column] == -99999).sum() > 10000:
        columns_to_be_removed.append(column)
 
# print(len(columns_to_be_removed))
cibil_df = cibil_df.drop(columns_to_be_removed, axis = 1)


# remove all the rows from each column that has a NULL value or -99999.
for column in cibil_df.columns:
    cibil_df = cibil_df[cibil_df[column] != -99999]
    

# checking for null values in both the dataset. 
internal_df.isna().sum()
cibil_df.isna().sum()

# Now we are merging both the dataframe(internal_df and cibil_df).

# firstly we check the common columns in both the dataframe.
for col1 in cibil_df.columns:
    for col2 in internal_df.columns: 
        if col1 == col2:
            print(col1)
            
# here below we merge both dataframe. here we are doing Inner join to avoid Null values.
final_merged_df = pd.merge(internal_df, cibil_df, how = 'inner', left_on = 'PROSPECTID', right_on = 'PROSPECTID')

# again check for null values.
final_merged_df.info()


## FEATURE SELECTION (CATEGORICAL THEN NUMERICAL).

# check or get the categorical columns. 
for column in final_merged_df.columns:
    if final_merged_df[column].dtype == 'object':
        print(column)
        
        
# checking the association of every categorical column with target column(Categorical).
for column in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2' , 'first_prod_enq2']:
    contingency_table = pd.crosstab(final_merged_df[column], final_merged_df['Approved_Flag'])
    chi_val, p_val, dof, expected = chi2_contingency(contingency_table)
    print(column, ' ----> ', p_val)
    
# Here we see that for each column the p_val is less than alpha(0.005), so each column has a association
# with Target column(Approved_Flag), we select the every column for further processing.


# check for the numerical columns and get them into a list.
numerical_columns = []
for column in final_merged_df.columns:
    if final_merged_df[column].dtype != 'object' and column not in ['PROSPECTID', 'Approved_Flag']:
        numerical_columns.append(column)
        
# print all the numerical columns.
print(numerical_columns) 
        
# calculate the value of VIF(Sequential VIF) to deal with the Multicolinarity between the numerical columns.
# VIF Sequentially checks.
# Best Method for Multicolinarity.

numeric_df = final_merged_df[numerical_columns]
total_numeric_columns = numeric_df.shape[1]
columns_to_be_kept = []
column_index = 0


# HERE WE PLOT A CORR MATRIX FOR VISUALIZATION.
# correlation matrix
corr_matrix = numeric_df.corr()


for i in range(0, total_numeric_columns):
    
    vif_value = variance_inflation_factor(numeric_df, column_index)
    print(numerical_columns[i], '-->', vif_value)
    
    if vif_value <= 6:
        columns_to_be_kept.append(numerical_columns[i])
        column_index = column_index + 1
    else:
        numeric_df = numeric_df.drop(numerical_columns[i], axis = 1)
        

# check for the Association for each numerical Column(using ANOVA becs here we have more than 2 columns).


final_numerical_columns_to_be_kept = []

for column in columns_to_be_kept:
    
    group_p1 = numeric_df[ final_merged_df['Approved_Flag'] == 'P1' ][column].dropna()
    group_p2 = numeric_df[ final_merged_df['Approved_Flag'] == 'P2' ][column].dropna()
    group_p3 = numeric_df[ final_merged_df['Approved_Flag'] == 'P3' ][column].dropna()
    group_p4 = numeric_df[ final_merged_df['Approved_Flag'] == 'P4' ][column].dropna()

    f_statistic, p_value = f_oneway(group_p1, group_p2, group_p3, group_p4)
    
    if p_value <= 0.05:
        final_numerical_columns_to_be_kept.append(column) 
        
        
# FEATURE SELECTION IS DONE FOR BOTH CATEGORICAL AND NUMERICAL COLUMNS.
        

# Now we list all the features.
features = final_numerical_columns_to_be_kept + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2' , 'first_prod_enq2']

final_merged_df = final_merged_df[features + ['Approved_Flag']]


# We do LabelEncoding or OneHotEncoding for categorical columns.
# ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2' , 'first_prod_enq2']

# Here Only one column that can be convert using LabelEncoding (EDUCATION has some order in categories).
# Rest of the columns are convert using OneHotEncoding.

print(final_merged_df['MARITALSTATUS'].unique())
print(final_merged_df['EDUCATION'].unique())
print(final_merged_df['GENDER'].unique())
print(final_merged_df['last_prod_enq2'].unique())
print(final_merged_df['first_prod_enq2'].unique())



# here we are doing LabelEncoding for EDUCATION.
# '12TH' 'GRADUATE' 'SSC' 'POST-GRADUATE' 'UNDER GRADUATE' 'OTHERS' 'PROFESSIONAL'

# 'SSC' = 1
# '12TH' = 2 
# 'GRADUATE' = 'UNDER GRADUATE' = 3 
# 'POST-GRADUATE' = 4
# 'PROFESSIONAL' = 5
# 'OTHERS' = 0 (Here they dont have a documents to proof his Education).


# ONE WAY to do it. (WE ALSO HAVE ANOTHER WAY OF DOING IT USING BUILT-IN OrdinalEncoder() in sklearn).
edu_mapping = {
    'OTHERS': 0,
    'SSC': 1,
    '12TH': 2,
    'UNDER GRADUATE': 3,
    'GRADUATE': 3,
    'POST-GRADUATE': 4,
    'PROFESSIONAL': 5
}

final_merged_df['EDUCATION_encoded'] = final_merged_df['EDUCATION'].map(edu_mapping)
final_merged_df['EDUCATION_encoded'] = final_merged_df['EDUCATION_encoded'].astype(int)
final_merged_df.info()

# Check count of each category in this column.
final_merged_df['EDUCATION_encoded'].value_counts()



# Now we need to do OneHotEncoding for rest of the columns.
final_encoded_df = pd.get_dummies(final_merged_df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2' , 'first_prod_enq2'])



# Here I drop a columns EDUCATION in after making a copy of original final_encoded_df.
final_encoded_df_copy = final_encoded_df


# drom a column EDUCATION from a final_encoded_df_copy.
final_encoded_df_copy = final_encoded_df_copy.drop('EDUCATION', axis = 1)


# MACHINE LEARNING MODEL FITTING/TRAINING.


# Data Preprocessing.
features = final_encoded_df_copy.drop(['Approved_Flag'], axis = 1)
target = final_encoded_df.loc[ : , 'Approved_Flag']



# train and test split.
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)



# 1. RANDOM FORREST ALGORITHM.

# model_train
rf_model = RandomForestClassifier(n_estimators = 200, max_depth = None, random_state = 42)
rf_model.fit(x_train, y_train)

# prediction on test data.
y_pred = rf_model.predict(x_test)

# Evaluation.
accuracy = accuracy_score(y_test, y_pred)
classification_Report = classification_report(y_test, y_pred)


# print accuracy and classification report. 
print('Accuracy of RandomForest : ', accuracy)
print('Classification Report : ', classification_Report)




# 2. XGBoost ALGORITHM.


# Encode labels to integers(fit on train only quki yaha humare pass multi-class prediction hai).
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)   # ['P1','P2','P3','P4'] -> [0,1,2,3]
y_test_enc = le.fit_transform(y_test)

# model training. 
n_classes = len(le.classes_)  # should be 4
xgb_model = XGBClassifier(objective = 'multi:softprob', num_class = n_classes)
xgb_model.fit(x_train, y_train_enc)

# prediction on the test data.
y_pred = xgb_model.predict(x_test)

# Evaluation.
accuracy = accuracy_score(y_test_enc, y_pred)
classification_Report = classification_report(y_test_enc, y_pred)

# print accuracy and classification report. 
print('Accuracy of XGBoost : ', accuracy)
print('Classification Report : ', classification_Report)




# 3. Decision Tree ALGORITHM.

# model training.
dec_model = DecisionTreeClassifier(max_depth = 20, min_samples_split = 10)
dec_model.fit(x_train, y_train_enc)

# prediction on the given test data. 
y_pred = dec_model.predict(x_test)

# Evaluation.
accuracy = accuracy_score(y_test_enc, y_pred)
classification_Report = classification_report(y_test_enc, y_pred)

# print accuracy and classification report. 
print('Accuracy of Decision Tree : ', accuracy)
print('Classification Report : ', classification_Report)

# Visualize the tree.
plt.figure(figsize=(12, 6))
plot_tree(dec_model, feature_names=features.columns, class_names=le.classes_, filled=True, max_depth=3)
plt.show()




# NOW WE FINETUNE THE XGBoost for further Study because it give me the best results.


# Standardization.

columns_to_be_scaled = ['Age_Oldest_TL','Age_Newest_TL','time_since_recent_payment',
                        'max_recent_level_of_deliq','recent_level_of_deliq',
                        'time_since_recent_enq','NETMONTHLYINCOME','Time_With_Curr_Empr']

# Do it using columnTransformer.

# create a ColumnTransformer.
preprocessing = ColumnTransformer(
    transformers = [
        ('num_scale', StandardScaler(), columns_to_be_scaled)
        ], 
    remainder = 'passthrough'   # keep the other columns remain unchange.
)


# apply the StandardScaler on the x_train and y_train dataset.
x_train_scaled = preprocessing.fit_transform(x_train)
x_test_scaled = preprocessing.fit_transform(x_test)


# Now fit the XGBoost model again on standardized data.

# Encode labels to integers(fit on train only quki yaha humare pass multi-class prediction hai).
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)   # ['P1','P2','P3','P4'] -> [0,1,2,3]
y_test_enc = le.fit_transform(y_test)

# model training.
n_classes = len(le.classes_)  # it should be 4.
xgb_model = XGBClassifier(objective = 'multi:softprob', num_class = n_classes)
xgb_model.fit(x_train_scaled, y_train_enc)


# prediction on testing data.
y_pred = xgb_model.predict(x_test_scaled)

# Evaluation.
accuracy = accuracy_score(y_test_enc, y_pred)
classification_Report = classification_report(y_test_enc, y_pred)

# print accuracy and classification report. 
print('Accuracy of XGBoost : ', accuracy)
print('Classification Report : ', classification_Report)

# NO IMPROVMENT OF STANDARDIZATION IN MATRICES (ACCURACY, RECALL, F1-SCORE).



# Now we come for doing HYPERPARAMETER TUNING in XGBoost Algorithm.

# Hyperparameter tuning in xgboost
from sklearn.model_selection import GridSearchCV

# Define the XGBClassifier with the initial set of hyperparameters
xgb_model = XGBClassifier(objective='multi:softmax', num_class=4)


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train_scaled, y_train_enc)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Best Hyperparameters: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}

# Evaluate the model with the best hyperparameters on the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test_scaled, y_test_enc)
print("Test Accuracy:", accuracy)



























