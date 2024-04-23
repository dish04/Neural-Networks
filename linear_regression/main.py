import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from LinearRegression import LinearRegressionRaw, coeff_determination

df = pd.read_csv('./archive/data1.csv')

print(df.columns)

forecast_col = 'selling_price'

df['label'] = df[forecast_col]

label_encoder = LabelEncoder()
df['name'] = label_encoder.fit_transform(df['name'])
df['fuel'] = label_encoder.fit_transform(df['fuel'])
df['seller_type'] = label_encoder.fit_transform(df['seller_type'])
df['transmission'] = label_encoder.fit_transform(df['transmission'])
df['owner'] = label_encoder.fit_transform(df['owner'])
#print(df.head())

x = np.array(df.drop(['label'], axis = 1))
y = np.array(df['label'])

x = preprocessing.scale(x)
y = np.array(df['label'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

clf1 = LinearRegression(n_jobs=-1)
y_hat1 = clf1.fit(x_train,y_train)
print(y_hat1.score(x_train,y_train))
