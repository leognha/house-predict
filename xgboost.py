import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

df_train = pd.read_csv(r"D:\users\leognha\Desktop\ML_final project\input\new_dataset_v2\train_data.csv")
df_test = pd.read_csv(r"D:\users\leognha\Desktop\ML_final project\input\new_dataset_v2\test_data.csv")
target = pd.read_csv("./train_house.csv")

#df_train = df_train.fillna(0)
#df_test  = df_test.fillna(0)
#target = df_train['total_price']

df_train = df_train.values
df_train = df_train[:, 1:]
#df_train = df_train.astype(np.float32)
df_test = df_test.values
#df_test = df_test[:, 1:]
target = target.values

#dtrain = xgb.DMatrix('train.csv?format=csv&label_column=234')
#dtest = xgb.DMatrix('test.csv?format=csv&label_column=233')

print(df_train.shape)
print(df_train.dtype)
print(target.dtype)

#train_X, test_X, train_y, test_y = train_test_split(df_train, target, test_size=0.2, random_state=7)


regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=1500,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

print('XGBRegressor')

regr.fit(df_train,target)
print('predic')
preds = regr.predict(df_test)

#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(dtrain, dtest)
#predictions = gbm.predict(test_X)

#model = XGBClassifier()
#model.fit(dtrain, dtest)

#y_pred = model.predict(X)

#idx = pd.read_csv(r"D:\users\leognha\Desktop\ML_final project\test.csv").building_id
#my_submissin = pd.DataFrame({ 'building_id': idx, 'total_price': y_te_RF })

print('1')
idx = pd.read_csv(r"D:\users\leognha\Desktop\ML_final project\test.csv").building_id
my_submissin = pd.DataFrame({ 'building_id': idx, 'total_price': preds })
my_submissin.to_csv(r"D:\users\leognha\Desktop\ML_final project\submission1500.csv", index=False)
print('1')
