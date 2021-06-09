"""
Three-layer neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a neural network for supervised learning problems with an input layer, a hidden layer and an output layer. The following can be varied:
* the number of nodes in the hidden layer
* the learning rate

The weights are initialized randomly with the maximum number of connections between pairs of layers.

This implementation assumes that the loss function is the sum of the squared differences between the target values and the output values, and that the activation function is the sigmoid. Any other choices would change the code for calculating the derivatives in the backpropagation.

Member methods:
* __init__
* sigmoid
* feedforward
* backpropagation
* calc_loss
"""

import numpy as np

MU = 0.0


class NeuralNetwork:
    def __init__(
        self,
        num_initial: int,
        num_hidden: int,
        num_final: int,
        learning_rate: float,
    ) -> None:
        self.num_initial = num_initial
        self.num_hidden = num_hidden
        self.num_final = num_final
        self.learning_rate = learning_rate

        # Initialize the outputs for each layer.
        self.initial = np.zeros(num_initial).reshape(-1, 1)
        self.hidden = np.zeros(num_hidden).reshape(-1, 1)
        self.final = np.zeros(num_final).reshape(-1, 1)

        # Initialize the weight matrices.
        # Use values normally distributed about mean mu and standard
        # deviation sigma calculated from 1/sqrt(n_incoming_nodes).
        sigma_init_hid = self.num_initial ** -0.5
        self.w_init_hid = np.random.normal(
            MU,
            sigma_init_hid,
            (self.num_hidden, self.num_initial)
        )
        sigma_hid_fin = self.num_final ** -0.5
        self.w_hid_fin = np.random.normal(
            MU,
            sigma_hid_fin,
            (self.num_final, self.num_hidden)
        )

    # Activation function
    def sigmoid(
        self,
        z: np.ndarray,
    ) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(
        self,
        example: np.ndarray,
    ) -> None:
        self.initial = example.copy()
        self.hidden = self.sigmoid(self.w_init_hid @ self.initial)
        self.final = self.sigmoid(self.w_hid_fin @ self.hidden)
        return

    def backpropagation(
        self,
        encoded_answer: np.ndarray
    ) -> None:
        diff = encoded_answer - self.final

        # Update the weights between the hidden and final layers. We're
        # calculating the following:
        # delta = 2 alpha[diff .* sigma'(final)] @ hidden.T
        delta = 2 * self.learning_rate * (
            diff * self.final * (1 - self.final) @ self.hidden.T
        )
        self.w_hid_fin += delta

        # Update the weights between the initial and hidden layers.
        # We're calculating the following:
        # [(1-hidden) .* diag(w_hid_fin.T @ delta)] @ initial.T
        self.w_init_hid += (
            (1 - self.hidden) * np.diag(
                self.w_hid_fin.T @ delta
            ).reshape(-1, 1)
        ) @ self.initial.T
        return

    # The loss function is the quadratic difference ||a-b||^2
    def calc_loss(
        self,
        encoded_answer: np.ndarray
    ) -> float:
        result = np.linalg.norm(self.final - encoded_answer) ** 2
        return result
