# Create a Neural Network for Two Category Classification with Keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# x < 50 = low
# x > 50 = high

# 10.5, 5, 9.5, 12 => low

model = Sequential()

# define neural network with 4 numbers...returns a numerical value representing the mean of the 4 #s
#
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# 0 = low
# 1 = high

# instead of a linear activation, we should use sigmoid,
# = a function that will return a single number between 0 and 1
#  closer it is to 0, the more likely the input is to be low.
#  The closer it is to 1, the more likely the input is to be high.
model.add(Dense(1, activation='sigmoid'))

#  change the loss from mean_squared_error to binary_crossentropy,
# = a loss function that will optimize the data for being a part of one of two classes
# + make a new metric to our model and call it accuracy, tells us the percentage of our training,
# validation, or test data points that were correctly identified.
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)

x_train = np.array([
  [1, 2, 3, 4],
  [4, 6, 1, 2],
  [10, 9, 10, 11],
  [101, 95, 89, 111],
  [99, 100, 101, 102],
  [105, 111, 109, 102]
])

# 0 = low
# 1 = high
# nstead of the mean of the four numbers, use the value 0 for low and 1
# for high. The first three inputs are 0, and the next three are 1.
y_train = np.array([
  [0],
  [0],
  [0],
  [1],
  [1],
  [1]
])

# define a validation set in the same way, then train the model with the
# fit method, using the same parameters that we would if we were only using a
#  linear output.
x_val = np.array([
  [1.5, 4, 3, 2.5],
  [10, 14, 11.5, 12],
  [111, 99, 105, 107]
])

y_val = np.array([
  [0],
  [0],
  [1],
])


model.fit(
  x_train,
  y_train,
  epochs=100,
  batch_size=2,
  verbose=1,
  validation_data=(x_val, y_val)
)


# run -> see 100 epochs of training, where we still see the loss value,
# but we also see accuracy for both training and validation. In just 100 epochs,
# we can get 100 percent accuracy on our small data set
