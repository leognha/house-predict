import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
import csv
#warnings.formatwarning('ignore')

train = pd.read_csv("./train.csv")
train = train.drop(labels=['building_id'], axis="columns")
train_label = pd.read_csv("./train.csv", usecols=[234])
train = train.drop(labels=['total_price'], axis="columns")

train = train.fillna(0)

test = pd.read_csv("./test.csv")
test = test.drop(labels=['building_id'], axis="columns")
test = test.fillna(0)

target = pd.read_csv("./train_house.csv")



model = RandomForestRegressor(n_estimators=1)
model.fit(train,train_label)
y_pred = model.predict(test)
np.savetxt("RandomForestRegressor_output", y_pred, delimiter=",")


