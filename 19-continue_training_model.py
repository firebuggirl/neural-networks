# Continue to Train an Already Trained Keras Model with New Data
# https://egghead.io/lessons/python-continue-to-train-an-already-trained-keras-model-with-new-data
from keras.models import load_model
import numpy as np
from keras.utils.np_utils import to_categorical

model = load_model('iris.h5')


# load summary of model to better understand trained model
model.summary()


#  make sure to shuffle the data again since we're going to be using
#  the validation_split method. Finally, we can call fit just like normal.



# import numpy and use that to load the iris.csv file
data = np.genfromtxt('iris.csv', delimiter=',')
data = np.random.permutation(data[1:, :])

#  need to remember to switch the y-values from just a number to a one-hot encoded vector.
#  First, import to_categorical and then we can use that on the y-values
x_train = data[:, :4]
y_train = to_categorical(data[:, 4])


model.fit(
  x_train,
  y_train,
  epochs=100,
  validation_split=0.2
)
