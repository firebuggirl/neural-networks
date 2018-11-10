
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# 10.5, 5, 9.5, 12 => 18.5

model = Sequential()

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

# can save the entire model definition w/
# the trained weights by using the model's 'save' method
# one argument = path to the file that you want to save
# use the h5 file extension because Keras uses the h5py library to
# make a binary file, but you can name file anything, like .model or .network.
# When we run now, we have a saved model file = a binary file that we can't
# read in a text editor, but it contains all the model information that we
# need to make predictions with the model, w/out requiring training again.
# run 'python 10-save_model.py'
model.save('mean_network.h5')
