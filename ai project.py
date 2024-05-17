#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import seaborn as sns 
from collections import Counter
from colorama import Fore, Style, init;
# Import necessary libraries
from IPython.display import display, HTML
from scipy.stats import skew  # Import the skew function
# Import Plotly.go
import plotly.graph_objects as go
# import Subplots
from plotly.subplots import make_subplots
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Model Train 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler , QuantileTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
# Classifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

import optuna

palette = ["#00203FFF", "#ADEFD1FF"]

color_palette = sns.color_palette(palette)
# Remove Warnings
import warnings 
warnings.filterwarnings("ignore")
# Set the option to display all columns
pd.set_option('display.max_columns', None)


# In[79]:


tr_d = pd.read_csv('creditcard_2023.csv')


# In[80]:


tr_d


# In[81]:


tr_d.drop('id',axis = 1 , inplace = True)


# In[82]:


tr_d


# In[84]:


def single_plot_distribution(column_name, dataframe):

    value_counts = dataframe[column_name].value_counts()

  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]}) 


    pie_colors = palette[0:3]
    ax1.pie(value_counts, autopct='%0.001f%%', startangle=90, pctdistance=0.85, colors=pie_colors, labels=None)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title(f'Distribution of {column_name}', fontsize=16)

    bar_colors = palette[0:3]
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax2, palette=bar_colors,) 
    ax2.set_title(f'Count of {column_name}', fontsize=16)
    ax2.set_xlabel(column_name, fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)


    ax2.tick_params(axis='x', rotation=45)


    plt.tight_layout()
    plt.show()


# In[85]:


# Class Ditribution
single_plot_distribution('Class',tr_d)


# In[33]:


columns_to_plot =['V1', 'V2', 'V3', 'V4','Class']

data=0
data_to_plot = data[columns_to_plot]


Q_colors = { 0 : palette[0], 1 : palette[1], 'other': 'gray'}  

sns.pairplot(data_to_plot, hue='Class', palette=Q_colors)
plt.show()


# In[86]:


def detect_outliers_iqr_with_visualization(dataframe, features):
    outlier_indices = []
    num_rows = 1
    num_cols = len(features)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))

    axes = axes.flatten()

    for i, feature in enumerate(features):

        Q1 = np.percentile(dataframe[feature], 25)
        Q3 = np.percentile(dataframe[feature], 75)
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = dataframe[(dataframe[feature] < Q1 - outlier_step) | (dataframe[feature] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

        sns.boxplot(x=dataframe[feature], ax=axes[i], color=palette[i % len(palette)])
        axes[i].set_title(f'Boxplot of {feature}')

    plt.tight_layout()
    plt.show()

    outlier_indices = [index for index, count in Counter(outlier_indices).items() if count > 2]
    
    return outlier_indices

columns_to_detect_outliers = ['V1', 'V2', 'V3', 'V4', 'V5', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

outlier_indices = detect_outliers_iqr_with_visualization(tr_d, columns_to_detect_outliers)

print("Indices of outliers:", outlier_indices)

tr_d_cleaned = tr_d.drop(outlier_indices, axis=0)

print("Shape of cleaned dataset after removing outliers:", tr_d_cleaned.shape)


# In[87]:


def plot_numerical_distribution_with_hue(data, num_cols, hue_col='Gender', figsize=(25, 25), dpi=100):
   
    rows = (len(num_cols) + 1) // 2 
    fig, ax = plt.subplots(rows, 2, figsize=figsize, dpi=dpi)
    ax = ax.flatten() 
      
    for i, column in enumerate(num_cols):  
        sns.histplot(data=data, x=column, hue=hue_col, ax=ax[i], kde=True, palette=palette)
        ax[i].set_title(f'{column} Distribution', size=14)
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(None)
        
        
        skewness = skew(data[column].dropna())
        skew_label = f'Skewness: {skewness:.2f}'
        
     
        ax[i].annotate(skew_label, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='red')
    
  
    for j in range(len(num_cols), len(ax)):
        fig.delaxes(ax[j])
    

    plt.tight_layout()
    
  
    plt.show()


# In[88]:


NUM_COLS_F = ['V1','V2','V3','V4','V5','V22','V23','V24','V25','V26','V27','V28']
plot_numerical_distribution_with_hue(tr_d,NUM_COLS_F,'Class')


# In[89]:


def apply_scaling(data, columns, scaler_type):
  
    if scaler_type == 'S':
        scaler = StandardScaler()  
    elif scaler_type == 'M':
        scaler = MinMaxScaler()  
    elif scaler_type == 'Q':
        scaler = QuantileTransformer(output_distribution='normal')  
    else:
        raise ValueError("Invalid scaler type. Choose 'S' for StandardScaler, 'M' for MinMaxScaler, or 'Q' for QuantileTransformer.")

   
    scaled_data = data.copy()

  
    for col in columns:
    
        scaled_data[col] = scaler.fit_transform(scaled_data[[col]])

    # Return the scaled data
    return scaled_data


# In[90]:


columns_to_scale =[col for col in tr_d.columns if tr_d[col].dtype == 'float']
scaler_type = 'M' 


tr_d = apply_scaling(tr_d, columns_to_scale, scaler_type)
print('Data Scaled Done')


# In[91]:


N_d = tr_d.select_dtypes(include='number')

correlation_matrix = N_d.corr()

plt.figure(figsize=(25, 15))
sns.heatmap(correlation_matrix, annot=True, cmap=palette, fmt=".1f", linewidths=0.5)
plt.title('Correlation Plot', fontsize=22)  
plt.tight_layout()  
plt.show()


# In[92]:


# # # =================================================================================================================
# # #                         X < y 
# # #================================================================================================================== 
X_T = tr_d.drop('Class', axis=1)
y_T = tr_d['Class']
# # # =================================================================================================================
# # #                         Train < Test Split
# # #================================================================================================================== 
X_TR, X_TE, Y_TR, Y_TE = train_test_split(X_T, y_T, test_size=0.1, random_state=42)

# # # =================================================================================================================
# # #                         Shapes < 
# # #================================================================================================================== 
print(f"Training set shape - X: {X_TR.shape}, y: {Y_TR.shape}")
print(f"Testing set shape - X: {X_TE.shape}, y: {Y_TE.shape}")


# In[93]:


# Initlize Models
# XGB Classifier
xgb = XGBClassifier(n_estimators=100, random_state=42)
# CatBoost Classifier
catboost = CatBoostClassifier(iterations=100, random_state=42 , verbose = 0)
# LightGBM Classifier
lgb_params = {
 'n_estimators': 890,
 'learning_rate': 0.7019434172842792,
 'max_depth': 19,
 'reg_alpha': 1.2057738033316066,
 'reg_lambda': 0.18598174484559382,
 'num_leaves': 3,
 'subsample': 0.746154395882518,
 'colsample_bytree': 0.3877680559022922
}
lgbm = LGBMClassifier(**lgb_params, random_state=42 , verbose = -1)
#___________________________________________________________________
print('Hurry ! Base Clfs Are Intilized')


# In[95]:


# 1. XGB Classifier
xgb.fit(X_TR ,Y_TR)
# Pred 
xgb_pred = xgb.predict(X_TE)
#-----------------------------
# 2. CatBoost Classifier
catboost.fit(X_TR, Y_TR)
# Pred
catboost_pred = catboost.predict(X_TE)
#-----------------------------
# 3. LightGBM Classifier
lgbm.fit(X_TR, Y_TR)
# Pred
lgbm_pred = lgbm.predict(X_TE)
#-----------------------------
print('Hurry ! Model Are Fitted')


# In[52]:


xgb_pred


# In[96]:


def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision,conf_matrix


# In[97]:


accuracy_XGB, precision_XGB, confusion_XGB = evaluate(Y_TE, xgb_pred)
print("Evaluation Results for XGB Classifier")
print(f"The Accuracy Score Of XGB Classifier is {accuracy_XGB}, Precision Is {precision_XGB},\nConfusion Matrix is \n{confusion_XGB} ")
print("\n")


accuracy_Cat, precision_Cat, confusion_Cat = evaluate(Y_TE, catboost_pred)
print("Evaluation Results for CatBoost Classifier")
print(f"The Accuracy Score Of CatBoost Classifier is {accuracy_Cat}, Precision Is {precision_Cat},\nConfusion Matrix is \n{confusion_Cat} ")
print("\n")

# LightGBM Classifier
accuracy_LGBM, precision_LGBM, confusion_LGBM = evaluate(Y_TE, lgbm_pred)
print("Evaluation Results for LightBoost Classifier")
print(f"The Accuracy Score Of CatBoost Classifier is {accuracy_LGBM}, Precision Is {precision_LGBM},\nConfusion Matrix is \n{confusion_Cat} ")
print("\n")


# In[98]:


evaluation_data = {
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'Accuracy': [accuracy_XGB, accuracy_LGBM, accuracy_Cat],
    'Precision': [precision_XGB, precision_LGBM, precision_Cat]
}


evaluation_df = pd.DataFrame(evaluation_data)


evaluation_df = evaluation_df.sort_values(by=['Accuracy', 'Precision'], ascending=False)


evaluation_df


# In[99]:


models = ['XGBoost', 'LightGBM', 'CatBoost']
accuracies = [accuracy_XGB, accuracy_LGBM, accuracy_Cat]
precisions = [precision_XGB, precision_LGBM, precision_Cat]

data = {'Model': models, 'Accuracy': accuracies, 'Precision': precisions}
df = pd.DataFrame(data)

custom_palette = sns.color_palette("Paired")

# Plot using Seaborn
plt.figure(figsize=(20, 8))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=df, palette=palette[0:2])
plt.title('Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# Subplot 2: Precision
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Precision', data=df, palette=palette[0:2])
plt.title('Model Precisions')
plt.xlabel('Model')
plt.ylabel('Precision')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[100]:


joblib.dump(xgb,'xgb_model_save.joblib')


# In[101]:


joblib.dump(catboost,'catboost_model_save.joblib')


# In[102]:


joblib.dump(lgbm,'lgbm_model_save.joblib')


# In[ ]:




