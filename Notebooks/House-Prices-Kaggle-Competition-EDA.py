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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # **House Prices Kaggle Competition - EDA**

# %% [markdown]
# ## **Data Dictionary**

# %% [markdown]
# MSSubClass: Identifies the type of dwelling involved in the sale.
#
#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
#
# MSZoning: Identifies the general zoning classification of the sale.
#
#        A	Agriculture
#        C	Commercial
#        FV	Floating Village Residential
#        I	Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park
#        RM	Residential Medium Density
#
# LotFrontage: Linear feet of street connected to property
#
# LotArea: Lot size in square feet
#
# Street: Type of road access to property
#
#        Grvl	Gravel
#        Pave	Paved
#
# Alley: Type of alley access to property
#
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
#
# LotShape: General shape of property
#
#        Reg	Regular
#        IR1	Slightly irregular
#        IR2	Moderately Irregular
#        IR3	Irregular
#        
# LandContour: Flatness of the property
#
#        Lvl	Near Flat/Level
#        Bnk	Banked - Quick and significant rise from street grade to building
#        HLS	Hillside - Significant slope from side to side
#        Low	Depression
#
# Utilities: Type of utilities available
#
#        AllPub	All public Utilities (E,G,W,& S)
#        NoSewr	Electricity, Gas, and Water (Septic Tank)
#        NoSeWa	Electricity and Gas Only
#        ELO	Electricity only
#
# LotConfig: Lot configuration
#
#        Inside	Inside lot
#        Corner	Corner lot
#        CulDSac	Cul-de-sac
#        FR2	Frontage on 2 sides of property
#        FR3	Frontage on 3 sides of property
#
# LandSlope: Slope of property
#
#        Gtl	Gentle slope
#        Mod	Moderate Slope
#        Sev	Severe Slope
#
# Neighborhood: Physical locations within Ames city limits
#
#        Blmngtn	Bloomington Heights
#        Blueste	Bluestem
#        BrDale	Briardale
#        BrkSide	Brookside
#        ClearCr	Clear Creek
#        CollgCr	College Creek
#        Crawfor	Crawford
#        Edwards	Edwards
#        Gilbert	Gilbert
#        IDOTRR	Iowa DOT and Rail Road
#        MeadowV	Meadow Village
#        Mitchel	Mitchell
#        Names	North Ames
#        NoRidge	Northridge
#        NPkVill	Northpark Villa
#        NridgHt	Northridge Heights
#        NWAmes	Northwest Ames
#        OldTown	Old Town
#        SWISU	South & West of Iowa State University
#        Sawyer	Sawyer
#        SawyerW	Sawyer West
#        Somerst	Somerset
#        StoneBr	Stone Brook
#        Timber	Timberland
#        Veenker	Veenker
#
# Condition1: Proximity to various conditions
#
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street
#        Norm	Normal
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
#
# Condition2: Proximity to various conditions (if more than one is present)
#
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street
#        Norm	Normal
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
#
# BldgType: Type of dwelling
#
#        1Fam	Single-family Detached
#        2FmCon	Two-family Conversion; originally built as one-family dwelling
#        Duplx	Duplex
#        TwnhsE	Townhouse End Unit
#        TwnhsI	Townhouse Inside Unit
#
# HouseStyle: Style of dwelling
#
#        1Story	One story
#        1.5Fin	One and one-half story: 2nd level finished
#        1.5Unf	One and one-half story: 2nd level unfinished
#        2Story	Two story
#        2.5Fin	Two and one-half story: 2nd level finished
#        2.5Unf	Two and one-half story: 2nd level unfinished
#        SFoyer	Split Foyer
#        SLvl	Split Level
#
# OverallQual: Rates the overall material and finish of the house
#
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
#
# OverallCond: Rates the overall condition of the house
#
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
#
# YearBuilt: Original construction date
#
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#
# RoofStyle: Type of roof
#
#        Flat	Flat
#        Gable	Gable
#        Gambrel	Gabrel (Barn)
#        Hip	Hip
#        Mansard	Mansard
#        Shed	Shed
#
# RoofMatl: Roof material
#
#        ClyTile	Clay or Tile
#        CompShg	Standard (Composite) Shingle
#        Membran	Membrane
#        Metal	Metal
#        Roll	Roll
#        Tar&Grv	Gravel & Tar
#        WdShake	Wood Shakes
#        WdShngl	Wood Shingles
#
# Exterior1st: Exterior covering on house
#
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
#
# Exterior2nd: Exterior covering on house (if more than one material)
#
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
#
# MasVnrType: Masonry veneer type
#
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
#
# MasVnrArea: Masonry veneer area in square feet
#
# ExterQual: Evaluates the quality of the material on the exterior
#
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
#
# ExterCond: Evaluates the present condition of the material on the exterior
#
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
#
# Foundation: Type of foundation
#
#        BrkTil	Brick & Tile
#        CBlock	Cinder Block
#        PConc	Poured Contrete
#        Slab	Slab
#        Stone	Stone
#        Wood	Wood
#
# BsmtQual: Evaluates the height of the basement
#
#        Ex	Excellent (100+ inches)
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
#
# BsmtCond: Evaluates the general condition of the basement
#
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
#
# BsmtExposure: Refers to walkout or garden level walls
#
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement
#
# BsmtFinType1: Rating of basement finished area
#
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
#
# BsmtFinSF1: Type 1 finished square feet
#
# BsmtFinType2: Rating of basement finished area (if multiple types)
#
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
#
# BsmtFinSF2: Type 2 finished square feet
#
# BsmtUnfSF: Unfinished square feet of basement area
#
# TotalBsmtSF: Total square feet of basement area
#
# Heating: Type of heating
#
#        Floor	Floor Furnace
#        GasA	Gas forced warm air furnace
#        GasW	Gas hot water or steam heat
#        Grav	Gravity furnace
#        OthW	Hot water or steam heat other than gas
#        Wall	Wall furnace
#
# HeatingQC: Heating quality and condition
#
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
#
# CentralAir: Central air conditioning
#
#        N	No
#        Y	Yes
#
# Electrical: Electrical system
#
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed
#
# 1stFlrSF: First Floor square feet
#
# 2ndFlrSF: Second floor square feet
#
# LowQualFinSF: Low quality finished square feet (all floors)
#
# GrLivArea: Above grade (ground) living area square feet
#
# BsmtFullBath: Basement full bathrooms
#
# BsmtHalfBath: Basement half bathrooms
#
# FullBath: Full bathrooms above grade
#
# HalfBath: Half baths above grade
#
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
#
# Kitchen: Kitchens above grade
#
# KitchenQual: Kitchen quality
#
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#
# Functional: Home functionality (Assume typical unless deductions are warranted)
#
#        Typ	Typical Functionality
#        Min1	Minor Deductions 1
#        Min2	Minor Deductions 2
#        Mod	Moderate Deductions
#        Maj1	Major Deductions 1
#        Maj2	Major Deductions 2
#        Sev	Severely Damaged
#        Sal	Salvage only
#
# Fireplaces: Number of fireplaces
#
# FireplaceQu: Fireplace quality
#
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
#
# GarageType: Garage location
#
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
#
# GarageYrBlt: Year garage was built
#
# GarageFinish: Interior finish of the garage
#
#        Fin	Finished
#        RFn	Rough Finished
#        Unf	Unfinished
#        NA	No Garage
#
# GarageCars: Size of garage in car capacity
#
# GarageArea: Size of garage in square feet
#
# GarageQual: Garage quality
#
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
#
# GarageCond: Garage condition
#
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
#
# PavedDrive: Paved driveway
#
#        Y	Paved
#        P	Partial Pavement
#        N	Dirt/Gravel
#
# WoodDeckSF: Wood deck area in square feet
#
# OpenPorchSF: Open porch area in square feet
#
# EnclosedPorch: Enclosed porch area in square feet
#
# 3SsnPorch: Three season porch area in square feet
#
# ScreenPorch: Screen porch area in square feet
#
# PoolArea: Pool area in square feet
#
# PoolQC: Pool quality
#
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
#
# Fence: Fence quality
#
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
#
# MiscFeature: Miscellaneous feature not covered in other categories
#
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
#
# MiscVal: $Value of miscellaneous feature
#
# MoSold: Month Sold (MM)
#
# YrSold: Year Sold (YYYY)
#
# SaleType: Type of sale
#
#        WD 	Warranty Deed - Conventional
#        CWD	Warranty Deed - Cash
#        VWD	Warranty Deed - VA Loan
#        New	Home just constructed and sold
#        COD	Court Officer Deed/Estate
#        Con	Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	Other
#
# SaleCondition: Condition of sale
#
#        Normal	Normal Sale
#        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand	Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit
#        Family	Sale between family members
#        Partial	Home was not completed when last assessed (associated with New Homes)
#

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
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import mlflow
import env_setup
import os

# %% [markdown]
# **Environment Setup**

# %%
result_path=env_setup.setup(repo_name='house-prices-kaggle-competition',nb_name='House-Prices-Kaggle-Competition-EDA')

# %% [markdown]
# **Read the data**

# %%
housing_data=pd.read_csv(os.path.join(result_path,'train.csv'))

# %%
housing_data.head()

# %%
housing_data.info()

# %% [markdown]
# ## **Exploratory Data Analysis**

# %% [markdown]
# **Check for missing values**

# %%
housing_data.isna().sum()

# %%
# Find out the columns with missing values
missing_value_cols=[col for col in housing_data.columns if housing_data[col].isna().sum()>0]
missing_values=[housing_data[col].isna().sum() for col in housing_data if housing_data[col].isna().sum()>0]
cols_with_missing_values=dict(zip(missing_value_cols,missing_values))
cols_with_missing_values

# %% [markdown]
# Columns like `Alley`,`MasVnrType`,`PoolQC`, `Fence`,`MiscFeature` can be dropped due to high percentage of missing values.

# %% [markdown]
# **Check for duplicated rows**

# %%
housing_data.duplicated().sum().sum()


# %% [markdown]
# **Utility Function to help with Visualization**

# %%
def visualize(plot:str,x:str,data:str,xlabel:str,y:str=None,hue:str=None,ylabel:str=None):
    '''
    '''
    fig=plt.figure(figsize=(10,5))

    if plot=='scatter' and y is not None:
        if hue is not None:
            sns.scatterplot(x=x,y=y,hue=hue,data=data)
        else:
            sns.scatterplot(x=x,y=y,data=data)
    elif plot=='bar' and y is not None:
        if hue is not None:
            sns.barplot(x=x,y=y,hue=hue,data=data)
        else:
            sns.barplot(x=x,y=y,data=data)
    elif plot=='box':
        sns.boxplot(x=x,data=data)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{xlabel} vs {ylabel}')
    plt.show()



# %% [markdown]
# **Utility function to find and remove outliers**

# %%
def find_and_remove_outliers(cols:list,data:str):
    '''
    '''
    X=data.copy()
    if all(item in X.columns for item in cols) is True and all(col in ['int32','int64','float32','float64'] for col in cols) is True:
        for col in cols:
            X_col_25=X[col].quantile(0.25)
            X_col_75=X[col].quantile(0.75)
            X_col_iqr=X_col_75-X_col_25
            X_col_lf=X_col_25-(1.5*X_col_iqr)
            X_col_uf=X_col_75+(1.5*X_col_iqr)
            X.loc[:,col]=X.loc[(X[col]>X_col_lf)&(X[col]<X_col_uf),col]
        return X



# %% [markdown]
# **Inspect `MasVnrArea` column**

# %%
# Box plot to check for outliers
visualize(plot='box',x='MasVnrArea',data=housing_data,xlabel='Masonry Veneer Area(sqft.)')

# %% [markdown]
# Since there are outliers, we can impute 0 for missing values instead of mean.

# %% [markdown]
# **Inspect `GarageYrBuilt` column**

# %%
# Box plot for GarageYrBuilt column
visualize(plot='box',x='GarageYrBlt',data=housing_data,xlabel='Garage Built Year')

# %% [markdown]
# Median value of year can be imputed for missing values.

# %% [markdown]
# **Inspect `MSSubClass` column**

# %% [markdown]
# From the data description, it seems `MSSubClass` column contains redundant information and does not offer any new information and can be dropped.

# %% [markdown]
# **Inspect `LotArea` column**

# %% [markdown]
# **Relation between `LotArea` and `SalePrice` with `LotConfig`**

# %%
# Scatter plot of LotArea and SalePrice with LotConfig
visualize(plot='scatter',x='LotArea',y='SalePrice',hue='LotConfig',data=housing_data,xlabel='Lot Area',ylabel='Sale Price')

# %% [markdown]
# `LotArea` column may be having outliers.

# %% [markdown]
# **Inspect `LotShape` column**

# %% [markdown]
# **Relation between `LotShape` and `SalePrice`**

# %%
# Bar plot of LotShape and SalePrice
visualize(plot='bar',x='LotShape',y='SalePrice',hue='LotShape',data=housing_data,xlabel='Lot Shape',ylabel='Sale Price')

# %% [markdown]
# Regular shaped houses are sold at less price than irregularly shaped houses.

# %% [markdown]
# **Inspect `LandContour` column**

# %%
# Bar plot of LandContour and SalePrice
visualize(plot='bar',x='LandContour',y='SalePrice',hue='LandContour',data=housing_data,xlabel='Land Contour',ylabel='Sale Price')

# %% [markdown]
# `LandContour` does not appear to be having any relationship with SalePrice.

# %% [markdown]
# **Inspect `LotConfig` column**

# %% [markdown]
# **Relation between `LotConfig` and `SalePrice`**

# %%
# Bar plot of LotConfig and SalePrice
visualize(plot='bar',x='LotConfig',y='SalePrice',hue='LotConfig',data=housing_data,xlabel='Lot Configuration',ylabel='Sale Price')

# %% [markdown]
# `LotConfig` does not appear to be directly contributing to the sale price.

# %% [markdown]
# **Inspect `Neighborhood` column**

# %% [markdown]
# **Relation between `Neighborhood` and `SalePrice`**

# %%
# Bar plot of Neighborhood and SalePrice
fig=plt.figure(figsize=(30,5))
sns.barplot(x='Neighborhood',y='SalePrice',hue='Neighborhood',data=housing_data)
plt.xlabel('Neighborhood')
plt.ylabel('Sale Price')
plt.title('Neighborhood vs SalePrice')
plt.show()

# %% [markdown]
# Neighborhoods are does not seem to be having strong relationship with price with most neighborhoods in same price range.

# %% [markdown]
# **Inspect `Condition1` and `Condition2` columns**

# %% [markdown]
# From data description, `Condition1` and `Condition2` columns can be dropped and better features can be created from its values.

# %% [markdown]
# **Split Predictors and Target columns**

# %%
y=housing_data['SalePrice']
X=housing_data.drop('SalePrice',axis=1)

# %%
cols_to_drop=['Id','Alley','MasVnrType','PoolQC','Fence','MiscFeature','MSSubClass']

# %% [markdown]
# **Split Training and Test data**

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)

# %%
X_train=X_train.drop(cols_to_drop,axis=1)

# %%
X_train.head()

# %%
X_train.info()

# %%
y_train.head()

# %% [markdown]
# **Impute missing values with median for numerical columns and mode for categorical columns**

# %%
num_cols=[col for col in X.columns if X[col].dtype in ['int32','int64','float32','float64'] and col not in cols_to_drop]
print(num_cols)
cat_cols=[col for col in X.columns if X[col].dtype=='object' and col not in cols_to_drop]
print(cat_cols)

# %%
num_imputer=SimpleImputer(strategy='median')
cat_imputer=SimpleImputer(strategy='most_frequent')

# %%
X_train[num_cols]=num_imputer.fit_transform(X_train[num_cols])
X_train[cat_cols]=cat_imputer.fit_transform(X_train[cat_cols])

# %%
X_train.head()

# %% [markdown]
# **Encode Categorical values**

# %%
ordinal=['LotShape','Utilities','LandSlope','HouseStyle','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','Electrical','KitchenQual','Functional',
'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive']

# %%
nominal=['MSZoning','Street','LandContour','LotConfig','Neighborhood','Condition1','Condition2',
'BldgType','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','Foundation','Heating','CentralAir','GarageType','SaleType','SaleCondition']

# %%
ordinal_encoder=OrdinalEncoder()
nominal_encoder=LabelEncoder()

# %% [markdown]
# **Encode Ordinal values**

# %%
X_train[ordinal]=ordinal_encoder.fit_transform(X_train[ordinal])

# %% [markdown]
# **Encode Nominal values**

# %%
for col in nominal:
    X_train[col]=nominal_encoder.fit_transform(X_train[col])

# %%
X_train.head()

# %% [markdown]
# **Mutual Information**

# %%
discrete_features=X_train.dtypes==int


# %%
def prep_mi_scores(X,y,discrete_features):
    mi_scores=mutual_info_regression(X=X,y=y,discrete_features=discrete_features)
    mi_scores=pd.Series(mi_scores,name='MI Scores',index=X.columns)
    mi_scores=mi_scores.sort_values(ascending=False)
    return mi_scores


# %%
mi_scores=round(prep_mi_scores(X=X_train,y=y_train,discrete_features=discrete_features),3)
print(mi_scores)

# %%
mi_scores.unique()

# %%
mi_scores[mi_scores>=0.2]
