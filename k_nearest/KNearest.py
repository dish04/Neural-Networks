import scipy
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import pandas as pd
import random


data = {
    'k': [[1, 2],[3,2],[2,1]],
    'r': [[6,7],[7,6],[8,7]]
    }

feature = [8,9]

#[[plt.scatter(i[0],i[1], color =j) for i in data[j]]for j in data]
#plt.scatter(feature[0],feature[1])
#plt.show()


def KNearest(data,predict ,k = 3):

    if len(data) >= k:
        warnings.warn('K is set to value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    #print(sorted(distances))
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

def score(data,x_test,y_test):
    correct = 0
    for i in range(len(x_test)):
        if y_test[i] == KNearest(data,x_test[i],k=5):
            correct += 1
    return correct/len(y_test)

KNearest(data,feature)