"""
MNIST handwriting classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script uses the classes Data, NeuralNetwork and Experiment to classify the MNIST data.

In this script the parameters that control the overall process of the experiment are set (the learning rate, number of hidden nodes and the number of epochs), as well as the choice of training and testing data sets.

To adapt this code for other experiments, note the following:
* The Data class is adapted to import and format the specific data used for this experiment, and would likely need to be extensively reworked.
* NeuralNetwork is a 3-layer fully-connected neural net. As long as the data format output from Data is the same, it should be able to work similarly. While it may look simple to adapt the activation function or loss function to other forms, any changes to these are likely to require a different calculation to be implemented by the backpropagation function, and may substantially complicate it. Changing which weights are non-zero may be useful for trying different structures to the 3-layer network. Changing the number of layers will require a substantially more complex rewrite.
* Experiment is a fairly generic procedure to run the required number of feedfoward and backpropagation steps and is likely to be easily adapted to different techniques. Improvements such as regularization and adaptive learning rates will likely be implemented in this class.
"""

import os
from mnist_experiment.data import Data
from mnist_experiment.experiment import Experiment
from mnist_experiment.neural_network import NeuralNetwork

# Where to find the data for training and testing.
TRAINING_DATA_LOC = "../data/mnist_train_small.csv"
TESTING_DATA_LOC = "../data/mnist_test_small.csv"

# The number of nodes in the hidden layer for this three-layer network.
NUM_HIDDEN = 100

# The fixed learning rate and number of epochs to use in this
# experiment's training phase.
LEARNING_RATE = .2
NUM_EPOCHS = 2

# Initialize the data sets, neural net and experiment.
directory = os.path.dirname(__file__)
testing_data = Data(
    directory=directory,
    data_file_loc=TESTING_DATA_LOC
)
training_data = Data(
    directory=directory,
    data_file_loc=TRAINING_DATA_LOC
)
neural_net = NeuralNetwork(
    num_initial=training_data.num_pixels,
    num_hidden=NUM_HIDDEN,
    num_final=training_data.num_targets,
    learning_rate=LEARNING_RATE
)
experiment = Experiment(
    num_epochs=NUM_EPOCHS,
    training_data=training_data,
    testing_data=testing_data,
    neural_net=neural_net
)

experiment.do_exp()
