# use high_low.csv to train neural network
# each row has 4 #s + a single output value of 0 or 1

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

# 10.5, 5, 9.5, 12 => low

# < 50 = lower
# > 50 = high
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)


# import the CSV
# data = a numpy array that contains all of the rows and columns that were in the CSV.
data = np.genfromtxt('high_low.csv', delimiter=',')


# split the data into the input values by taking all the rows except for the first,
# because that's the header row, and the first four column which are inputs.
# Then, we can extract the output class values by taking all the rows except
# for the header again, and only the fourth column this time which is the
#  output value.
x_train = data[1:, :4]
y_train = data[1:, 4]

# use the x_train and y_train values to fit the model,
# passing them into the X and Y arguments, specifying 100 epoch and
# a validations split of 20 percent
model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)
