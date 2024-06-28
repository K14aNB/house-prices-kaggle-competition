# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: house-prices-kaggle-competition
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **House Prices Kaggle Competition - Pipeline**

# %% [markdown]
# ## **Initial Setup**

# %% [markdown]
# **Check and install the dependencies**

# %%
# !curl -sSL https://raw.githubusercontent.com/K14aNB/house-prices-kaggle-competition/main/requirements.txt

# %%
# %%capture
# Run this command in terminal before running this notebook as .py script
# Installs dependencies from requirements.txt present in the repo
# !pip install -r https://raw.githubusercontent.com/K14aNB/house-prices-kaggle-competition/main/requirements.txt

# %% [markdown]
# **Import the libraries**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
import mlflow
import env_setup
import pickle
import os

# %% [markdown]
# **Environment Setup**

# %%
result_path=env_setup.setup(repo_name='house-prices-kaggle-competition',nb_name='House-Prices-Kaggle-Competition-Pipeline')

# %% [markdown]
# **Read the data**

# %%
housing_data=pd.read_csv(os.path.join(result_path,'train.csv'))

# %%
housing_data.head()

# %%
housing_data.info()

# %% [markdown]
# **Split Predictors and Target**

# %%
y=housing_data['SalePrice']
X=housing_data.drop('SalePrice',axis=1)

# %%
params={
    'mutual_information_lower_limit':0.2,
    'num_impute_strategy':'median',
    'num_fill_value':'NA',
    'cat_impute_strategy':'most_frequent',
    'cv_folds':5
}

# %% [markdown]
# **Retrieve the Mutual Information**

# %%
with open('mi_scores.pkl','rb') as pkl:
    mi_series=pickle.load(pkl)
    print(mi_series)

# %%
# Selecting columns with mi_score>=0.2
columns_baseline=list(mi_series[mi_series>=params.get('mutual_information_lower_limit')].index.values)
print(columns_baseline)

# %% [markdown]
# ## **Preprocessing**

# %%
num_cols=[col for col in X.columns  if col in columns_baseline and X[col].dtype in ['int32','int64','float32','float64']]
print(num_cols)
cat_cols=[col for col in X.columns if col in columns_baseline and X[col].dtype=='object']
print(cat_cols)

# %% [markdown]
# **Impute missing values for numerical columns**

# %%
if params.get('numerical_impute_strategy')=='constant':
    numerical_imputer=SimpleImputer(strategy=params.get('numerical_impute_strategy',fill_value=params.get('num_fill_value')))
else:
    numerical_imputer=SimpleImputer(strategy=params.get('num_impute_strategy'))

# %% [markdown]
# **Numerical Pipeline Steps**

# %%
numerical_pipeline=Pipeline(steps=[
    ('numerical_imputer',numerical_imputer),
    ('scaler',RobustScaler())
])

# %% [markdown]
# **Impute missing values for categorical columns**

# %%
categorical_imputer=SimpleImputer(strategy=params.get('cat_impute_strategy'))

# %% [markdown]
# **Encode categorical variables**

# %%
ordinal=['ExterQual','BsmtQual','KitchenQual','GarageFinish']

# %%
nominal=['Neighborhood','Foundation']

# %% [markdown]
# **Ordinal Encoder for encoding ordinal variables**

# %%
cat_ordinal_encoder=OrdinalEncoder()


# %% [markdown]
# **Custom Transformer for encoding nominal variables**

# %%
class NominalEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_=pd.DataFrame(X,columns=self.columns)
        le=LabelEncoder()
        for col in self.columns:
            X_[col]=le.fit_transform(X_[col])
        return X_
    
    def get_feature_names_out(self,*args):
        return self.columns



# %%
# Initialize the Custom Transformer NominalEncoder
cat_nominal_encoder=NominalEncoder(columns=nominal)

# %% [markdown]
# **One-Hot Encoder for encoding nominal variables**

# %%
ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False)

# %% [markdown]
# **Categorical Ordinal Pipeline**

# %%
cat_ordinal_pipeline=Pipeline(steps=[
    ('categorical_imputer',categorical_imputer),
    ('ordinal_encoder',cat_ordinal_encoder),
])

# %% [markdown]
# **Categorical Nominal Pipeline**

# %%
cat_nominal_pipeline=Pipeline(steps=[
    ('categorical_imputer',categorical_imputer),
    ('nominal_encoder',ohe)
])

# %% [markdown]
# ### **Preprocessing Transformer**

# %%
preprocessor=ColumnTransformer(transformers=[
    ('numerical_transformer',numerical_pipeline,num_cols),
    ('categorical_ordinal_transformer',cat_ordinal_pipeline,ordinal),
    ('categorical_nominal_transformer',cat_nominal_pipeline,nominal)
])


# %% [markdown]
# ## **Features**

# %% [markdown]
# ### **Feature Pipeline**

# %% [markdown]
# **Feature Selection**

# %%
class SelectFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,columns):
        self.columns=columns
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X[:,self.columns]
             
    def get_feature_names_out(self,*args):
        return self.columns
    


# %% [markdown]
# **Linear Regression**

# %%
lr_run_name='Linear Regression_CV'
lr=LinearRegression()

# %% [markdown]
# ## **Final Pipeline**

# %%
pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',lr)
])

# %% [markdown]
# ## **Cross Validation**

# %%
cv_scores=cross_validate(pipeline,X,y,scoring=['r2','neg_root_mean_squared_error'],cv=params.get('cv_folds'),error_score='raise')

# %% [markdown]
# **Model Metrics**

# %%
cv_scores

# %%
# R2
avg_r2=round(cv_scores['test_r2'].mean(),3)
print(avg_r2)

# RMSE
avg_rmse=round(-1.0*cv_scores['test_neg_root_mean_squared_error'].mean(),3)
print(avg_rmse)

# %% [markdown]
# ## **Log the Model in MLFlow** 

# %%
with mlflow.start_run(run_name=lr_run_name):
    pipeline.fit(X,y)
    mlflow.log_params(params)
    mlflow.log_metrics({'r2_score':avg_r2,'rmse':avg_rmse})
    signature=mlflow.models.infer_signature(X,y,params)
    mlflow.sklearn.log_model(sk_model=pipeline,artifact_path=lr_run_name,signature=signature,registered_model_name=lr_run_name)

# %% [markdown]
#
