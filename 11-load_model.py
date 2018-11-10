from keras.models import load_model

import numpy as np

# pass file name of the model file that we saved
# result = entire trained model
model = load_model('mean_network.h5')

# make a few input arrays to make predictions on
x_predict = np.array([
  [1.5, 2, 3.5, 4],
  [13, 11, 9, 14],
  [102, 98.5, 102.5, 100]
])

output = model.predict(x_predict)

print("")
print("Expected: 2.75, 11.75, 100.75")
print("Actual: ", output)


# see model details.. see all of the layers in the network and their sizes
# can load and inpsect pre-saved models ...can use this fully trained model to make predictions
# model.summary()
