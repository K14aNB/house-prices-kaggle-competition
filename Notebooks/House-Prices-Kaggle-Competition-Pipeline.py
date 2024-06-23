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
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import env_setup
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

# %%
params={
    'num_impute_strategy':'median',
    'num_fill_value':'NA',
    'cat_impute_strategy':'most_frequent'
}

# %% [markdown]
# ## **Preprocessing**

# %%
cols_to_drop=['Id','Alley','MasVnrType','PoolQC','Fence','MiscFeature','MSSubClass']

# %%
num_cols=[col for col in housing_data.columns if col not in cols_to_drop and housing_data[col].dtype in ['int32','int64','float32','float64']]
print(num_cols)
cat_cols=[col for col in housing_data.columns if col not in cols_to_drop and housing_data[col].dtype=='object']
print(cat_cols)

# %% [markdown]
# **Impute missing values for numerical columns**

# %%
if params.get('numerical_impute_strategy')=='constant':
    numerical_imputer=SimpleImputer(strategy=params.get('numerical_impute_strategy',fill_value=params.get('num_fill_value')))
else:
    numerical_imputer=SimpleImputer(strategy=params.get('num_impute_strategy'))

# %% [markdown]
# **Impute missing values for categorical columns**

# %%
categorical_imputer=SimpleImputer(strategy=params.get('cat_impute_strategy'))

# %% [markdown]
# **Impute Transformer**

# %%
imputer=ColumnTransformer(transformers=[
    ('numerical_imputer',numerical_imputer,num_cols),
    ('categorical_imputer',categorical_imputer,cat_cols)
],remainder='passthrough')

# %% [markdown]
# **Encode categorical variables**

# %%
ordinal=['LotShape','Utilities','LandSlope','HouseStyle','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','Electrical','KitchenQual','Functional',
'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive']

# %%
nominal=['MSZoning','Street','LandContour','LotConfig','Neighborhood','Condition1','Condition2',
'BldgType','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','Heating','CentralAir','GarageType','SaleType','SaleCondition']

# %%
cat_ordinal_encoder=OrdinalEncoder()

# %%
cat_nominal_encoder=LabelEncoder()

# %% [markdown]
# **Encoder Transformer**

# %%
encoder=ColumnTransformer(transformers=[
    ('ordinal_encoder',cat_ordinal_encoder,ordinal),
    ('nominal_encoder',cat_nominal_encoder,nominal)
],remainder='passthrough')

# %% [markdown]
# **Preprocessing Pipeline**

# %%
preprocessor=Pipeline(steps=[
    ('imputer',imputer),
    ('encoder',encoder)
])

# %%
