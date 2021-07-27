[MNIST](https://en.wikipedia.org/wiki/MNIST_database) is a dataset containing 70,000 28x28 pixel images of handwritten digits (0–9).
This project implements a fully-connected three-layer neural network that classifies the MNIST data by identifying which digit each image represents. This implementation performs surprisingly well, correctly classifying around 95% of the images with the right choices of hyperparameters; nothing like the state of the art but pretty impressive for something so basic!

The goal is to show in very simple terms how an elementary neural network is put together; but "simple" here means "as simple as possible but no simpler". Every detail is included and explained on the assumption that the reader knows python, numpy and only the most basic background on how neural nets work.

Outline
-------

Neural network experiments generally have three main components:
* **Data:** the training and testing data, which will need to be loaded and formatted
* **Neural network:** the number of layers and the weights connecting them, the choices of activation and loss functions; these will be connected via the feedforward and backpropagation functions that define the model
* **Experiment:** how the data and model are used in training and testing cycles

These components will each be created as its own class to be combined in a simple script.

Acknowledgments
---------------

I started this project while reading of [Tariq Rashid](https://makeyourownneuralnetwork.blogspot.com/)'s *Make Your Own Neural Network*. This book explains everything the author is doing in explicit detail that is very helpful for a detail-oriented learner like myself. The basic choices made in setting this problem up, the mathematical details for the backpropagation calculation, and some aspects of the code are based on that book.

Resources
---------

* This code uses standard libraries like numpy but no high-powered libraries like PyTorch or TensorFlow.
* The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) was created by Yann LeCun, Corinna Cortes and Christopher J.C. Burges.
* However, I used a simpler-to-deal-with [CSV version](https://pjreddie.com/projects/mnist-in-csv/) of it that was provided by Joseph Redmon and recommended in Rashid's book.

Instructions
------------

* This code was written using Python 3.9 and some standard libraries such as numpy; no machine learning specific libraries such as PyTorch were used.
* Due to the file-size limits of github I've only included small versions of the training and testing data, which should work for trying out the code. If you want to run the code with the full dataset, download the CSV files linked above and change the TRAINING_DATA_LOC and TESTING_DATA_LOC variables to the correct file names.
* To try out different versions of the experiment change the hyperparameters (LEARNING_RATE, NUM_EPOCHS, NUM_HIDDEN) in main.py. They are currently set to some generic values that run quickly for testing.

```
pip install numpy
python3 src/main.py
```
