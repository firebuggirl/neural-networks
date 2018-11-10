
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# 10.5, 5, 9.5, 12 => 18.5

model = Sequential()

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')


# get a summary of the neural network
# returns a summary of all of network layers + a number that represents
# the total trainable weights in the network
# more weights = a more complex function + longer training time + more data needed
# can change network by modifying # of nodes in a layer or by modifying the
# network entirely. EX: less layers
model.summary()
