from keras.models import Sequential
from keras.layers import Dense

import numpy as np

model = Sequential()

# 10.5, 5, 9.5, 12 => 18.5
# model takes 4 numbers as inputs....ie., inpute_dim=4
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(
  optimizer='adam',
  loss='mean_squared_error'
)

# Define array of inputs w/ input examples, each an array containing 4 #s
x_train = np.array([
  [1, 2, 3, 4],
  [4, 6, 1, 2],
  [10, 9, 10, 11],
  [10, 12, 9, 13],
  [99, 100, 101, 102],
  [105, 111, 109, 102]
])
# Define array of outputs
# # OUtput y values = the mean of each of the rows from the x inputs
y_train = np.array([
  [2.5],
  [3.25],
  [10.0],
  [11.0],
  [100.5],
  [106.75]
])


# train the model
# because we only have 6 input data points, should pick a batch size that is smaller in number
# 'epoch' =  how many times the network will loop through the entire data set..more = network accuracy, but longer time to train
# 'verbose=1'= allow us to see the loss at every epoch.
model.fit(
  x_train,
  y_train,
  batch_size=2,
  epochs=100,
  verbose=1
)
