https://egghead.io/lessons/python-course-introduction-fully-connected-neural-networks-with-keras

		- create primary network -> configure -> test accuracy

		- save model -> learn to load model -> make predictions

		- expose the model as a web app

# Create a Fully Connected TensorFlow Neural Network with Keras

		- Install Python

		- Install Tensorflow = backend

		- Install Keras = api/Python library

		- You may also need h5py if you don't already have it

		- Configure Keras to use Tensorflow

		Check installations:

			` python --version `

			` pip install tensorflow `//install tensorflow

			` pip show tensorflow `//get/show version

			https://www.tensorflow.org/

			` sudo pip install keras `

			https://keras.io/#installation

			"Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation"

			` pip install h5py `//already have it

			` python -c 'import keras; print(keras.__version__)' `//output should = 'Using TensorFlow backend.'

			` touch neural_net.py `

			- `Sequential` = a linear stack of layers between the input and output...ie., no loops or extra in/out nodes

			- `Dense` = model...network layer

			- `Dense(8,...` = # of hidden nodes

				- ..common to have a network that increased in size towards middle, then shrinks back down towards


				- last (4th) layer = output layer ...1st 3 layers are hidden


				` model.compile(
				  optimizer='adam',//many optimizers to choose from...can check out other optimizers in future
				  loss='mean_squared_error'//also many loss functions to choose from
				)`

				- trying to get as close as we can to a particular number as our output.. use the common `mean_squared_error` Loss function




# Train a Sequential Keras Model with Sample Data


https://egghead.io/lessons/python-train-a-sequential-keras-model-with-sample-data


   - ` touch train_sample_data.py `

	 - run:

	 		` python train_sample_data.py  `


		-  a fairly `low loss` = the mean squared error between the actual Y values and the predicted values from our network


# Separate Training and Validation Data Automatically in Keras with validation_split

https://egghead.io/lessons/python-separate-training-and-validation-data-automatically-in-keras-with-validation_split

https://github.com/nanohop/keras_neural_network/blob/validation-split/validation_split.py

	` touch validation_split.py `

# Manually Set Validation Data While Training a Keras Model

https://egghead.io/lessons/python-manually-set-validation-data-while-training-a-keras-model

https://github.com/nanohop/keras_neural_network/blob/validation-data/validation_data.py


	-  some cases where you don’t want an automatic validation sample - but you want to be able to provide your own validation data set.

	   EX: `time series data` -> using a sequential set of data to do validation



	- ` touch validation_data.py `


	- ` touch antarctica_weather.csv `



# Evaluate a Keras Model with Test Data

https://egghead.io/lessons/python-evaluate-a-keras-model-with-test-data

https://github.com/nanohop/keras_neural_network/tree/evaluate-model


	- ` touch evaluate_model.py `


	- Once we have a model with good training + validation accuracy -> create test  data + evaluate the model’s accuracy on data via `evaluate` method on the `Keras model`


	- ` python evaluate_model.py `


			- see the hundred epochs like normal with our training loss and validation loss.

			- We also see a single run with the `three-new test data points`. Finally, we see that our metric is loss. This value represents the loss on the as yet unseen data, which is the test set that we created.



# Testing Different Neural Network Topologies

https://egghead.io/lessons/python-testing-different-neural-network-topologies

- investigate a few different typical network topologies including adding more “depth” and “width”, and evaluate what network topology is best for our data set

		- ` touch deep_network.py `

				- ` python deep_network.py `

		- ` touch wide_shallow_network.py `

				- also effective, but remember that dataset is small, increase + re-test

				` python wide_shallow_network.py `



# Understand the Structure of a Keras Model by Viewing the Model Summary

https://egghead.io/lessons/python-understand-the-structure-of-a-keras-model-by-viewing-the-model-summary

https://github.com/nanohop/keras_neural_network/tree/model-summary

	` touch model_summary.py `

	- print a summary of the neural network with the `summary` method on the Keras model

	- prints a summary w/ all of the network layers + a number that represents the total trainable weights in the network

					- more `weights` = more complexity + longer training time + need for larger data set

	- can change the neural network by modifying the number of nodes in a layer, or by modifying the network entirely, to get a better feel for how complex the network is, and how many trainable weights it contains.


# Make Predictions on New Data with a Trained Keras Models

https://egghead.io/lessons/python-make-predictions-on-new-data-with-a-trained-keras-models

	- use your network to make predictions on data in production

	- ` touch predict.py `

	- Keras models trained w/ the `fit` method can be used to make predictions via the `predict` method


# Save a Trained Keras Model Weights and Topology to a File

https://egghead.io/lessons/python-save-a-trained-keras-model-weights-and-topology-to-a-file


https://egghead.io/lessons/python-save-a-trained-keras-model-weights-and-topology-to-a-file



		- Instead of training the network every single time it is run,  save the Keras model for use in the future. Via Keras -> save both the model weights + topology to a single file

		- `  touch save_model.py `

		- ` python save_model.py `

					- creates `mean_network.h5`...a binary file


# Load and Use a Saved Keras Model

https://egghead.io/lessons/python-load-and-use-a-saved-keras-model

https://github.com/nanohop/keras_neural_network/tree/load-model


		- load a saved model from a file, and then use it to make predictions on new data

		- ` touch load_model.py `

		-  import load model from keras.models

				- ` from keras.models import load_model `

				- ` import numpy as np `

						- make a few input arrays to make predictions on + use the model's predict method to make predictions

						- ` python load_model.py  `//see the model's output predictions via `Actual`

# Create a Neural Network for Two Category Classification with Keras

https://egghead.io/lessons/python-create-a-neural-network-for-two-category-classification-with-keras

https://github.com/nanohop/keras_neural_network/tree/binary-classification

		- take a Keras network designed for `continuous (linear) output`, and convert it into a network for `binary classification`, which can `divide data into two classes` (for example: “dog” vs “cat”), and can be used for things like `sentiment analysis` ("positive" vs "negative").

		- have a neural network defined -> takes in four numbers -> returns a numerical value that = the mean of those four numbers. Instead, what if we wanted to classify our data with the network?

				- EX:

						- < 50 = `low` class

					  - 	> 50 = `high` class

						- everything else in network remains the same, just the `output` is different

		- ` touch binary_classification.py `

		- ` python binary_classification.py `


# Import Data From a CSV to Use with a Keras Model Using NumPy’s genfromtxt Method

https://egghead.io/lessons/python-import-data-from-a-csv-to-use-with-a-keras-model-using-numpy-s-genfromtxt-method

https://github.com/nanohop/keras_neural_network/tree/import-csv


	- replace our set of sample data with data that we import from a CSV, by importing it as a numpy array using numpy's `genfromtxt method` -> use that data to train the binary classification model.

			- `touch high_low.csv `

					- contains a hundred rows of inputs to train our neural network. Each row has four numbers + a single output value of a `0` or `1`

			- ` touch import_csv.py `

			- ` python import_csv.py `


# Make Binary Class Predictions with Keras Using predict and predict_classes


https://egghead.io/lessons/python-make-binary-class-predictions-with-keras-using-predict-and-predict_classes


https://github.com/nanohop/keras_neural_network/tree/predict-classes

		- use the Keras model’s `predict` method to look at the predicted class value. Then use the `predict_classes` method to have Keras make a class prediction and return only a 0 or a 1 -> = the predicted class

	  - ` touch predict_classes.py `

		- ` python predict_classes.py `


# Create a Dense Neural Network for Multi Category Classification with Keras

https://egghead.io/lessons/python-create-a-dense-neural-network-for-multi-category-classification-with-keras

https://egghead.io/lessons/python-create-a-dense-neural-network-for-multi-category-classification-with-keras


		- take a network set up for binary classification, and turn it into a network that can take 3 or more classes. This network will let us go beyond classifying data into only two categories, and will allow us to expand to any number of categories (for example: “dog” vs “cat” vs “mouse”). Then, we’ll use real data from a csv to train and test that network.

		` touch muti_class_model.py `

		` touch iris.csv `

		- training the neural networks on a CSV with two classes, 0 for low and 1 for high. Now, we're going to switch that to a new dataset


		- `Iris` dataset = a common training set used to test neural networks

				- Each row represents a different flower and each flower has four data points, the Sepal Length, Sepal Width, Petal Length, and Petal Width.

				- the final column is the `class of the flower`, which is a `0`, a `1`, or `2`. A 0 represents Iris setosa, a 1 is Iris versicolor, and 2 is Iris virginica.

				- have to `convert the network` from a binary classification network to a `multi-class classification` network

				- can't use `binary_crossentropy` anymore -> use `categorical_crossentropy (can use any number of classes)`

						- HOWEVER, can't take up the flower classes as just a number like 0, 1, or 2, but instead, it needs to have the class represented by a `one-hot encoded vector`.

								- Keras has a built-in function to do that translation for us:


										- ` from keras.utils.np_utils import to_categorical `



										- watch/re-read!!!!!!


# Make Predictions on New Data with a Multi Category Classification Network

https://egghead.io/lessons/python-make-predictions-on-new-data-with-a-multi-category-classification-network

https://github.com/nanohop/keras_neural_network/tree/multi-class-prediction

			- Once we have built a multi-class classification network, we'll use it to make predictions on new data that wasn't used during training. We'll start by calling the `predict` method, which returns the probability that each input data row belongs to each one of the possible classes. Then, we'll use the `predict_classes` method to only output the class prediction as an integer, which might be easier to use in a production system.

			- ` touch multi_class_prediction.py `


# Change the Learning Rate of the Adam Optimizer on a Keras Network

https://egghead.io/lessons/python-change-the-learning-rate-of-the-adam-optimizer-on-a-keras-network

https://github.com/nanohop/keras_neural_network/tree/change-learning-rate


		-  can specify several options on a network optimizer, like the learning rate and decay, so we’ll investigate what effect those have on training time and accuracy. Each data sets may respond differently, so it’s important to try different optimizer settings to find one that properly trades off training time vs accuracy for your data.


		- ` touch change_learning_rate.py `

		- using the `adam` optimizer for the network which has a default learning rate of `.001`. To change that, first import Adam from keras.optimizers:

				` from keras.optimizers import Adam `

				- create a new instance of the Adam optimizer, and use that instead of a string to set the optimizer


# Change the Optimizer Learning Rate During Keras Model Training

https://egghead.io/lessons/python-change-the-optimizer-learning-rate-during-keras-model-training


https://github.com/nanohop/keras_neural_network/tree/change-lr-and-keep-training


			- break our training up into multiple steps, and use different learning rates at each step. This will allow the model to train more quickly at the beginning by taking larger steps, but we will reduce the learning rate in later steps, in order to more finely tune the model as it approaches an optimal solution. If we just used a high learning rate during the entire training process, then the network may never converge on a good solution, and if we use a low learning rate for the entire process, then the network would take far too long to train. Varying the learning rate gives us the best of both worlds (high accuracy, with a fast training time).

			- setting the learning rate for the Adam optimizer before we fit, but we may want to change that later and retrain with a lower learning rate.


			- ` touch iris.h5 `

			- ` touch change_lr_and_keep_training.py `


# Continue to Train an Already Trained Keras Model with New Data

https://egghead.io/lessons/python-continue-to-train-an-already-trained-keras-model-with-new-data

https://github.com/nanohop/keras_neural_network/tree/continue-training-model

					- As we get new data, we will want to re-train our old models with that new data. We’ll look at how to load the existing model, and train it with new data, and then save the newly trained model.

					- ` touch continue_training_model.py `

					- can load an existing model by importing load_model from keras.models, and then call load_model and pass the file name of our saved model. We can look at the summary of that model to better understand what we just loaded.


					` from keras.models import load_model
						import numpy as np
						from keras.utils.np_utils import to_categorical

						model = load_model('iris.h5')

						model.summary() `

						- can also continue training the saved model if we want to. We don't have the model defined in this file at all, but the saved file contains all the information that we need to pick up training right where we left off. We can import numpy and use that to load the iris.csv file.

						- `  python continue_training_model.py `

						- model will pick up just where it left off and continue to train the network. This can be especially helpful if you get new data and you want to train an old model, or if you simply want to pause training because it's taking a long time and resume at a later time.
