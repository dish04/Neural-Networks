import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from KNearest import KNearest
from KNearest import score
data = pd.read_csv('./cancer_data_set/wdbc.data')
data = data.drop(['ID'] ,axis=1)

labeler = LabelEncoder()
skle = labeler.fit_transform(data['Diagnosis'])
#print(skle)
data['Diagnosis'] = skle

x_train,x_test,y_train,y_test = train_test_split(data.drop(['Diagnosis'] ,axis=1) ,data['Diagnosis'] ,test_size=0.1)

data2 = {
    0:[],
    1:[]
}

#print(x_train)
for rows in data.loc[:250,:].values:
    data2[rows[0]].append(rows[1:])

predict = np.array([10.29,27.61,65.67,321.4,0.0903,0.07658,0.05999,0.02738,0.1593,0.06127,0.2199,2.239,1.437,14.46,0.01205,0.02736,0.04804,0.01721,0.01843,0.004938,10.84,34.91,69.57,357.6,0.1384,0.171,0.2,0.09127,0.2226,0.08283])


#Prediction for predict
model1 = KNeighborsClassifier()
model1.fit(x_train,y_train)
model1_prediction = model1.predict([predict])

model2_prediction = KNearest(data2,predict,k=10)
# print(y_test)
print("SciPy's classifier's prediction = ",model1_prediction[0],"\nRaw KNN's prediction = ",model2_prediction)
print("SciPy's classifier's score = ",model1.score(x_test,y_test),"\nRaw KNN's score = ",score(data2,np.array(x_test),np.array(y_test)))


