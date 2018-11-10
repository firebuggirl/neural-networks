from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

import numpy as np

model = Sequential()


# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris-virginica
# 'softmax' = ensure that all three probabilities add to 1, which will help our
# neural network decide which class it might belong to then we can run our model.
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

# import the iris.csv file
data = np.genfromtxt('iris.csv', delimiter=',')


# 'to_categorical' = have the class represented by a
# one-hot encoded vector instead of a number
# have an array of one-hot encoded vectors, which means the index 0 value is a 1,
#  if the class was a 0, the index 1 value is a 1, if the class was a 1, and the index 2 value is a 1, if the class was a 2.


x_train = data[1:, :4]
y_train = to_categorical(data[1:, 4])

# shuffle the data before we do a fit
perm = np.random.permutation(y_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]

model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)
