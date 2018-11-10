# Change the Learning Rate of the Adam Optimizer on a Keras Network
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
model.add(Dense(3, activation='softmax'))

# create a new instance of the Adam optimizer, and use that instead of
# a string to set the optimizer.
# lr = learning rate
# If the learning rate is too small, the network will never have a chance
# to get where it's going. The accuracy will always be low, or training
#  will take a really long time.
#  ..if too large, then the network will jump all over the place and will
#  never be able to find the best solution, because it will keep jumping over it
opt = Adam(lr=0.001)
#After 100 epoch, we're seeing good progress on the training in validation accuracy,
#so keep that learning rate. Important to test different learning rates
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

model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)
