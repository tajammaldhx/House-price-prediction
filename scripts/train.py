"""Reproduce the notebook Ridge workflow; export preprocessing and weights for JS."""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parents[1]
train=pd.read_csv(ROOT/'data/train.csv').drop(columns='Id')
test=pd.read_csv(ROOT/'data/test.csv').drop(columns='Id')
train=train[train.GrLivArea<4500].copy()
y=np.log1p(train.SalePrice).reset_index(drop=True)
raw=pd.concat([train.drop(columns='SalePrice'),test],ignore_index=True).drop(columns=['PoolQC','MiscFeature','Alley','Fence'])
for c in ['MSSubClass','MoSold','YrSold']:raw[c]=raw[c].astype(str)
features=raw.copy()
def filling_null(features):
    
    # In the description for 'Functional Feature', it is given that Typ refers to the Typical Functionality
    # We will replace null values in the 'Functional' feature with 'Typ'
    
    features['Functional'] = features['Functional'].fillna('Typ')
    
    # Replace the null values in the below columns with their mode. We are replacing with mode because
    # all of these features are categorical variable, we can't take the mean nor the median.
    
    features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])
    
    features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
    
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    
    # In order to fill the null values for MSZoing, we will take into account the MSSubClass featue.
    # We are doing so because the type of dwelling in a given area largely affects the zone of the area.
    
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    # In order to fill the null values for LotFrontage(Linear feet of street connected to property) 
    # we will fill it with median of the values grouped by Neighborhood.
    
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    # As mentioned before that the 'NA' value in many columns represent the absence of that feature for
    # a particular house, we will replace those null values with None.
    # 'None' for categorical feature and 0 for numerical feature.
    
    cat_features = features.select_dtypes(include = 'object').columns
    num_features = features.select_dtypes(exclude = 'object').columns
    
    features[cat_features] = features[cat_features].fillna('None')
    features[num_features] = features[num_features].fillna(0)
        
    return features
def new_features(features):
    
    features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x : 1 if x > 0 else 0)
    features['HasGarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    features['HasBsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    # Adding total sqfootage features 
    # In order to get the total area of the house we have to add the basement area, 1st floor area and 2nd floor area.
    
    features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
    
    # Total number of bathrooms = number of full bathrooms + 0.5 * number of half bathrooms
    # + number of bathrooms in basement + 0.5 * number of half bathrooms in the basement.
    
    features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                                   features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))
    
    features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                                  features['EnclosedPorch'] + features['ScreenPorch'])

    # Add years since remodel
    features['YearsSinceRemodel'] = features['YrSold'].astype(int) - features['YearRemodAdd'].astype(int)

    return features
features=filling_null(features)
base=features.copy()
schema=[]
for c in base.columns:
    if base[c].dtype=='object':
        schema.append({'name':c,'type':'category','options':sorted(base[c].unique().tolist())})
    else:
        schema.append({'name':c,'type':'number','min':float(base[c].min()),'max':float(base[c].max())})
def record(row):
    return {k:(str(v) if k in ['MSSubClass','MoSold','YrSold'] else v) for k,v in row.items()}
examples=[record(base.iloc[i].to_dict()) for i in [1460,1461,1462]]
features=new_features(features)
categorical=features.select_dtypes(include='object').columns.tolist()
lambdas={}
for c in features.select_dtypes(exclude='object').columns:
    # Binary indicators stay untransformed: Pearson Box-Cox optimization is unstable.
    if features[c].nunique()>2 and abs(features[c].skew())>0.5:
        try:
            lambdas[c]=float(boxcox_normmax(features[c]+1))
        except Exception:
            print('Bounded Box-Cox optimizer used for',c)
            lambdas[c]=float(boxcox_normmax(features[c]+1,optimizer=lambda f:minimize_scalar(f,bounds=(-5,5),method='bounded')))
        features[c]=boxcox1p(features[c],lambdas[c])
encoded=pd.get_dummies(features)
X=encoded.iloc[:len(y)]
itr,ite=train_test_split(np.arange(len(y)),test_size=.25,random_state=0)
model=Ridge(alpha=.1).fit(X.iloc[itr],y.iloc[itr])
pred=model.predict(X.iloc[ite])
metrics={'logRMSE':float(np.sqrt(mean_squared_error(y.iloc[ite],pred))),'dollarMAE':float(mean_absolute_error(np.expm1(y.iloc[ite]),np.expm1(pred))),'train':len(itr),'test':len(ite),'features':len(encoded.columns),'records':len(y)}
# Explicit feature mapping avoids ambiguous underscores in column names.
mapping=[]
for c in encoded.columns:
    match=next(((k,c[len(k)+1:]) for k in categorical if c.startswith(k+'_')),None)
    mapping.append({'category':match[0],'value':match[1]} if match else {'numeric':c})
result={'schema':schema,'defaults':examples[0],'examples':examples,'lambdas':lambdas,'mapping':mapping,'coefficients':model.coef_.tolist(),'intercept':float(model.intercept_),'metrics':metrics}
(ROOT/'dist/model.json').write_text(json.dumps(result,allow_nan=False))
(ROOT/'validation.json').write_text(json.dumps({'rows':[record(base.iloc[int(i)].to_dict()) for i in ite],'predictions':np.expm1(pred).tolist()}))
pd.DataFrame({'Id':pd.read_csv(ROOT/'data/test.csv').Id,'SalePrice':np.expm1(model.predict(encoded.iloc[len(y):]))}).to_csv(ROOT/'predictions.csv',index=False)
print(json.dumps(metrics,indent=2))
