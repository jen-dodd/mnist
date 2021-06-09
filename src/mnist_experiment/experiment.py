"""
Neural network experiment
~~~~~~~~~~~~~~~~~~~~~~~~~
Sets up everything needed to run a full supervised learning experiment with a neural network. Requires to be passed in:
* the desired number of epochs;
* a neural net initialized with the desired graph, starting parameters, learning rate, activation function, and loss function;
* training and testing data.

Member methods:
* __init__
* train
* calc_score
* test
* do_exp
"""

import numpy as np
from mnist_experiment.data import Data
from mnist_experiment.neural_network import NeuralNetwork


class Experiment:
    def __init__(
        self,
        num_epochs: int,
        training_data: Data,
        testing_data: Data,
        neural_net: NeuralNetwork
    ) -> None:
        self.num_epochs = num_epochs
        self.training_data = training_data
        self.testing_data = testing_data
        self.net = neural_net

        # We'll record the performance for each testing example and
        # calculate an overall fraction correct as the final metric.
        self.scores = np.array([])
        self.fraction_correct = 0

    def train(
        self,
    ) -> None:
        for epoch in range(self.num_epochs):
            for i in range(self.training_data.num_images):
                image = self.training_data.images[i].reshape(-1, 1)
                encoded_answer = self.training_data.encoded_answers[i] \
                    .reshape(-1, 1)
                self.net.feedforward(image)
                self.net.backpropagation(encoded_answer)

    def calc_score(
        self,
        encoded_answer: np.ndarray
    ) -> int:
        target_losses = np.zeros(len(self.testing_data.targets)).reshape(-1, 1)
        for i in range(len(self.testing_data.targets)):
            target = self.testing_data.targets[i].reshape(-1, 1)
            target_losses[i] = self.net.calc_loss(target)
        guess = np.argmin(target_losses)
        if guess == np.argmax(encoded_answer):
            return 1
        return 0

    def test(
        self
    ) -> None:
        for i in range(self.testing_data.num_images):
            image = self.testing_data.images[i].reshape(-1, 1)
            self.net.feedforward(image)
            encoded_answer = self.testing_data.encoded_answers[i].reshape(-1, 1)
            score = self.calc_score(encoded_answer)
            self.scores = np.append(self.scores, score)
            if i > 9990:
                print("Target answer:\n" + str(encoded_answer))
                print("Output answer:\n " + str(self.net.final))
                print("Score is: " + str(score))
        self.fraction_correct = np.average(self.scores)
        return

    def do_exp(
        self,
    ) -> None:
        self.train()
        self.test()
        print(
            "The fraction that were identified correctly is: " + str(self.fraction_correct)
        )
