
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/ ()

# 1 - Load Data

    - 1st, set random number seed:

          - ` numpy.random.seed(7) `

          - Pima Indians onset of diabetes dataset

                - a standard machine learning dataset from the UCI Machine Learning repository

                - a binary classification problem (onset of diabetes as 1 or not as 0)

                - Dataset details:

                  https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names

          - can now load the file directly using the NumPy function `loadtxt()`

          -  ...`8` input variables and `1` output variable (the last column)


# Define Neural Network Model


    - Models in Keras = sequence of layers

          - add layers one at a time

          - ensure the input layer has the right number of inputs via `input_dim`

          - use a fully-connected network structure with three layers

                - defined using `Dense` class

                - `first layer` has 12 neurons and expects 8 input variables.

                - `second hidden layer` has 8 neurons

                - `output layer` has 1 neuron to predict the class (onset of diabetes or not)


# Compile Model

    - training a network = finding the best set of weights to make predictions for this problem

    - must specify the loss function to use to evaluate a set of weights


      - This EX uses ` logarithmic loss`...ie., `binary_crossentropy`

      - also use the gradient descent algorithm `adam` = default

            - learn more: https://arxiv.org/abs/1412.6980


# Fit Model


        - execute the model on some data


        - train or fitmodel on loaded data by calling `fit()` function on the model

# Evaluate Model        


    - could separate your data into train and test datasets for training and evaluation of your model



# Run

      ` python keras_first_network.py `


          - should see a message for each of the 150 epochs printing the loss and accuracy for each, followed by the final evaluation of the trained model on the training dataset.
