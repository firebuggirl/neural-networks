# Create a Dense Neural Network for Multi Category Classification with Keras
# Iris dataset is a common dataset used to test neural networks
# each row represents a flower + each flower has 4 data points
# final column = the class of the flower
# https://egghead.io/lessons/python-create-a-dense-neural-network-for-multi-category-classification-with-keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

import numpy as np

model = Sequential()


# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris-virginica

# 'softmax' -> ensures that all 3 probabilities add to one
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# There are now 3 classes = need to convert network from a binary classification network
# to a multi-class classification network w/ 'categorical_crossentropy'
# cannot accept the flower classes as a number (...0, 1, 2..etc...), instead
# class is represented by a 'one hot encoded vector' -> use/import 'to_categorical' to do translation
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

data = np.genfromtxt('iris.csv', delimiter=',')

x_train = data[1:, :4]
y_train = to_categorical(data[1:, 4])

perm = np.random.permutation(y_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]

model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)

# make some new data by making a numpy array with three new data rows.
# This is new data that the model hasn't seen yet.
# = one new data row for each class of data that we have
predict_data = np.array([
  [4.9, 3.0, 1.5, 0.2], #0 Iris-setosa
  [5.7, 3.0, 4.5, 1.2], #1 Iris-versicolor
  [7.2, 3.2, 6.4, 2.3]  #2 Iris-virginica
])

# make a class prediction for each row
# If we add a line to call predict_classes, and rerun that, we can see that the
# output is just the class numbers that are being predicted, which match up with
# the one hot encoded output that we saw before, but it may be easier to interpret and use.
output = model.predict(predict_data)

# Tell numpy to suppress scientific notation, by calling set_printoptions on numpy,
# + set suppress equal to True...Makes the output is easier to read.
np.set_printoptions(suppress=True)

# To interpret this output, remember that the input was made into categorical
# data by calling to_categorical with the data, which turns it from an
# integer like zero, one, or two into a one hot encoded value.

# print the output with a blank line for formatting
print("")
print(output)

# OUtput = Each row is the row of our input data, and each column represents one
# of the possible classes.
output = model.predict_classes(predict_data)

print("")
print(output)
