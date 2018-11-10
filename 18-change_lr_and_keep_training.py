# Change the Optimizer Learning Rate During Keras Model Training

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

model = Sequential()

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# set rate of Adam learning optimizer before we fit..try different rates...ie. 0.005, etc....
opt = Adam(lr=0.001)

model.compile(
  optimizer=opt,
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris-virginica

data = np.genfromtxt('iris.csv', delimiter=',')

x_train = data[1:, :4]
y_train = to_categorical(data[1:, 4])

perm = np.random.permutation(y_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]

# After we fit the first time, we can change the model optimizer by setting
# model.optimizer to a new Adam optimizer with a lower learning rate.
# Then we can call fit again with the same parameters as before.
model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)

model.optimizer = Adam(lr=0.0001)
# by first writing with a high learning rate and then switching to a small
# learning rate is telling the network that it can start by taking large steps,
# which gives it more freedom to explore the training landscape.
# Then when we want to start refining the results, without risking taking a big
#  step in the wrong direction, we lower the learning rate and continue training.

# when we run that it starts with a hundred epochs at the first learning rate
# and then continues with another hundred epochs at the smaller learning rate.

model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)

# save model so that we can reload the fully-trained model later
model.save('iris.h5')
