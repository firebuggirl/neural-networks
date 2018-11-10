# Sequential = a linear stack of layers between the input and output
from keras.models import Sequential
# Dense = model...network layer
from keras.layers import Dense

model = Sequential()

# 10.5, 5, 9.5, 12 => 18.5
# relu = rectified linear unit, ... most commonly used activation function y = max(0, x)
# it is linear for all positive values and = zero for all negative values
# it is sparsely activated = better predictive power and less overfitting/noise
# gets rid of data that is irrelevant
# Good article explaining relu: https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(
  optimizer='adam',
  loss='mean_squared_error'
)
