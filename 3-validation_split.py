from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# 10.5, 5, 9.5, 12 => 18.5

model = Sequential()

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

x_train = np.array([
  [1, 2, 3, 4],
  [4, 6, 1, 2],
  [10, 9, 10, 11],
  [10, 12, 9, 13],
  [99, 100, 101, 102],
  [105, 111, 109, 102],
  [3, 7, 4, 1],
  [1, 8, 3, 7],
  [12, 15, 11, 9],
  [9, 15, 10, 11],
  [110, 99, 105, 101],
  [97, 101, 100, 105]
])

y_train = np.array([
  [2.5],
  [3.25],
  [10.0],
  [11.0],
  [100.5],
  [106.75],
  [3.75],
  [4.75],
  [11.75],
  [11.25],
  [103.75],
  [100.75]
])


# perm = permutation of numbers that match the size of the output array = act
# of arranging all the members of a set into some sequence or order
# gives us a random array of array indices that we can use to reset
# the order of our X and Y arrays
perm = np.random.permutation(y_train.size)
x_train = x_train[perm]
y_train = y_train[perm]


# 'validation_split' = a decimal between 0 & 1 ...represents the
# percentage of the training data to use as the validation data set
#  split into 20 percent validation and 80 percent training for this example
#  when using Keras's automatic validation split : always takes the last X percent of
#  the data that you give it, so that if you have ordered data of any kind, you may want to shuffle your data before training
model.fit(
  x_train,
  y_train,
  epochs=100,
  batch_size=2,
  verbose=1,
  validation_split=0.2
)


# retrain the network with the validation set..there is an extra 'val_loss'
# output value = the mean squared error loss on the validation set.
# key thing to look for, though, is that the loss and the validation loss are
# more consistent as you increase the amount of data that you give the network.
#
#
