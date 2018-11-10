from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
# 9 columns and the range 0:8 will select columns from 0 to 7,
# stopping before index 8.
# How to Index, Slice and Reshape NumPy Arrays for Machine Learning in Python:
# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
# run for a small number of iterations (150) and use a relatively small batch
# size of 10....can be chosen experimentally by trial and error
# where the work happens on your CPU or GPU...No GPU is required for this example
# BUT,  how to run large models on GPU hardware cheaply in the cloud:
# https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/
model.fit(X, Y, epochs=150, batch_size=10)


# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
