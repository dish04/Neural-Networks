import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2D Linear Regression
def _best_fit_slope_and_intercept(data):
    x = data[0]
    y = data[1]
    m = ((np.mean(x)*np.mean(y))-np.mean(x*y))/(np.mean(x)**2 - np.mean(x**2))
    b = np.mean(y) - np.mean(x)*m
    return m,b

def coeff_determination(data,y_hat):
    se_y_hat = ((data[1]-y_hat)**2).sum()
    se_y_mean = ((data[1]-np.mean(data[1]))**2).sum()
    return 1-(se_y_hat/se_y_mean)

def LinearRegressionRaw(data):
    #Raw Linear Regression
    m,b = _best_fit_slope_and_intercept(data)
    #y_hat is best fit line
    y_hat = m*data[0] + b
    return y_hat

if __name__ == "__main__":
    x= np.linspace(1,10,10)
    y = [5,7,10,11,15,13,20,23,17,20]
    data = np.array([x,y])
    #plt.scatter(data[0],data[1])
    line = LinearRegressionRaw(data)
    #plt.plot(data[0],line)
    print(coeff_determination(data,line))
    plt.show()