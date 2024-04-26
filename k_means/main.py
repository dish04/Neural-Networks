import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn import datasets

data = pd.read_excel('titanic.xls')

# print(data.head())

# print(data.columns)
data = data.drop('name',axis=1)
data = data.drop('body',axis=1)
data = data.drop('boat',axis=1)

encoder = LabelEncoder()
data['sex'] = encoder.fit_transform(data['sex'])
data['embarked'] = encoder.fit_transform(data['embarked'])
data['cabin'] = encoder.fit_transform(data['cabin'])
data['home.dest'] = encoder.fit_transform(data['home.dest'])

for i in range(len(data['ticket'])):
    if type(data['ticket'][i]) == int:
        # print(data['ticket'][i])
        data.loc[i,'ticket'] = str(i)
        
data['ticket'] = encoder.fit_transform(data['ticket'])
data.replace(np.nan,-99999)
data.fillna(-99999,inplace=True)

x_train,x_test,y_train,y_test = train_test_split(data.drop('survived',axis=1),data['survived'],test_size=0.3)
print(np.nan in x_train)
# [print(data[i].dtype," ",i) for i in data.columns]
# print(data['ticket'])
clf = KMeans(n_clusters=3)
clf.fit(x_train)



# clf = KMeans(n_clusters=3)
# clf.fit(X)

# centroids = clf.cluster_centers_
# labels = clf.labels_

# [plt.scatter(i[0],i[1], c= 'r') for i in X]
# plt.scatter(centroids[:,0], centroids[:,1],marker='x',c = 'orange')
# plt.show()
# #print(X,y)