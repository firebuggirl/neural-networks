
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# 10.5, 5, 9.5, 12 => 18.5

model = Sequential()

# define a single output via 'linear' -> get one numeric value in each output array
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

x_train = np.array([
  [1, 2, 3, 4],
  [4, 6, 1, 2],
  [10, 9, 10, 11],
  [10, 12, 9, 13],
  [99, 100, 101, 102],
  [105, 111, 109, 102]
])

y_train = np.array([
  [2.5],
  [3.25],
  [10.0],
  [11.0],
  [100.5],
  [106.75]
])


x_val = np.array([
  [1.5, 4, 3, 2.5],
  [10, 14, 11.5, 12],
  [111, 99, 105, 107]
])

y_val = np.array([
  [2.75],
  [11.875],
  [105.5],
])


model.fit(
  x_train,
  y_train,
  epochs=100,
  batch_size=2,
  verbose=1,
  validation_data=(x_val, y_val)
)


# Predict
# only required parameter to predict is the input values
# shapes of those values need to be numpy arrays that match
# the training input data
# Then can add one or more set of inputs that we want to make predictions for
#  predict method can take more than one input data at the same time
x_predict = np.array([
  [1.5, 2, 3.5, 4],
  [13, 11, 9, 14],
  [102, 98.5, 102.5, 100]
])

#
output = model.predict(x_predict)

# The mean of all four inputs, 1.5, 2, 3.5, and 4, is 2.75. We'll print out
# that we expect to see 2.75, and we'll print the actual value that the
# network predicts.
print("")
print("Expected: 2.75, 11.75, 100.75")
print("Actual: ", output)
