import numpy as npy
import pandas as pan
from matplotlib import pyplot as plt

#Without using tf or other tools

#Using mnist data set to understand the basics of neural networks

data = pan.read_csv('train.csv')
data = npy.array(data)
m, n = data.shape
npy.random.shuffle(data)

# shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

#Setting initial params

def init_params():
    W1 = npy.random.randn(10,784)
    b1 = npy.random.randn(10,1)
    W2 = npy.random.randn(10,10)
    b2 = npy.random.randn(10,1)
    return W1 ,b1, W2, b2

#Functions required in forward propagation
def Rel_U(Z):
    return npy.maximum(Z,0)

def softmax(Z):
    A = npy.exp(Z)/ sum(npy.exp(Z))
    return A

#Forward porpagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = Rel_U(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#Functions required in backward propagation
def one_hot(Y):
    one_hot_Y = npy.zeros((Y.size, Y.max() + 1))
    one_hot_Y[npy.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def RelU_deriv(Z):
    return Z > 0

#Back propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    yed = one_hot(Y)
    dZ2 = A2-yed
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * npy.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * RelU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * npy.sum(dZ1)
    return dW1, db1, dW2, db2

#Updating weights and biases
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return npy.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return npy.sum(predictions == Y) / Y.size

#Gradient decent aka the learning part to minimise the cost function
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy",get_accuracy(predictions, Y))
    return W1, b1, W2, b2

#Pressing the start training button
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

#Testing our model
test_prediction(5,W1,b1,W2,b2)
