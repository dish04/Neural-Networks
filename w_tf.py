import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test ) = tf.keras.datasets.mnist.load_data()

#Normalizing the values from 0 to 255 to 0 to 1 (axis is to mention along which axis normalization will occur)
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_train,axis=1)

#Setting the model
model = tf.keras.models.Sequential()

#Adding layers to our sequencial model
model.add(tf.keras.layers.Flatten()) #Flattens input data from 28*28 to a single long array
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #Adding first hidden layer which is using 128 neurons and ReL_U as activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #Adding second hidden layer which is using 128 neurons and ReL_U as activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #Adding the output layer which is 10 long for each output from 0 to 9 using softmax to squish the values ranging between 0 and 1

model.compile(
    optimizer='adam', #The most basic optimizer used
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #Can use different loss functions depending on the setup or the predictor
    metrics=['sparse_categorical_accuracy']#Wacthing the accuracy
)

#Running the model
history = model.fit(x_train, y_train , epochs=5)
print(history)

#Saving our model
model.save("mnist_model_using_tf")

#Loading model to predict
loaded_model = tf.keras.models.load_model('mnist_model_using_tf')
predictions = loaded_model.predict([x_test])
print(np.argmax(predictions[3]))

plt.imshow(x_test[3])
plt.show()