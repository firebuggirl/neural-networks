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

# 0 = class = 'low'
# 1 = class = 'high'
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)

data = np.genfromtxt('high_low.csv', delimiter=',')

x_train = data[1:, :4]
y_train = data[1:, 4]

# train network
model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)
#neural network takes in four numerical values and predicts a class of 0, if the
# values are low, and 1, if the values are high. After the neural network is
# trained, make some inputs to test that are numpy arrays of four values each.
x_predict = np.array([
  [10, 25, 14, 9],
  [102, 100, 75, 90]
])

# have Keras round these predicted values (ie., the raw output from the sigmoid function
# that we defined on the output node) to get to a class,
output = model.predict_classes(x_predict)

print("")
print(output)


#  run 'python predict_classes.py' ->
# network returns a floating point value & not a class of '0' or '1' without the predict_classes method
# change to predict_classes method to see classes of '0' or '1' returned
