import numpy as np
import matplotlib.pyplot as plot 
import matplotlib.pylab as plt
from math import sqrt


class NeuralNetwork:
    """
    Neural network class

    define your fully connected network easily 
    and train it for your dataset.
    """

    def __init__(self, structure, activations):
        """
        Object initialization

        :param structure: define your layers as a list of number of nodes, [n_inputs, hidden_layers, n_output]
                Example : [20, 10, 10, 5]

        :param activations: activation function for each layer. len(activations) = len(layers) - 1
                Example ['sig', 'sig', 'tanh']
        """

        self.activations = activations

        self.weights = []
        self.biases = []

        # weight initialization
        for i in range(len(structure) - 1):
            if self.activations[i] == 'sig' or self.activations[i] == 'tanh':
                # Normalized Xavier
                value = sqrt(6.0) / sqrt(structure[i] + structure[i + 1])

            else:
                # Menhaj
                value = 3 / sqrt(structure[i + 1])

            # generate random numbers
            self.weights.append(np.random.uniform(low=(-1 * value), high=value, size=(structure[i + 1], structure[i])))
            self.biases.append(np.random.uniform(low=(-1 * value), high=value, size=(structure[i + 1], 1)))

            print(f"layer {i} : W = {self.weights[i].shape} B = {self.biases[i].shape}")

        # define activation functions as lambda
        self.sig = lambda x: 1 / (1 + np.exp(-1 * x))
        self.tanh = lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        self.linear = lambda x: x
        self.relu = lambda x: x if x >= 0 else 0

    def learn(self, inputs, labels, epochs=100, learning_rate=0.01):
        """
        Train neural network

        :param inputs: list of your datas. this MUST be a python list, NOT numpy array or anything else.
                Example : [ [data 1] , [data 2] ]
        :param labels: list of your labeks. this MUST be a python list, NOT numpy array or anything else.
                Example : [ [label for data 1] , [label for data 2] ]
        :param epochs: number of training iterations
        :param learning_rate: the learning rate

        :return loss: list of losses at every epoch
        """

        loss = []
        for e in range(1, epochs + 1):
            epoch_loss = []

            for x, y in zip(inputs, labels):
                
                # labels preprocess 
                Y = np.array(y)
                Y = Y.reshape(Y.shape[0], 1)

                # inputs preprocess 
                X = np.array(x)
                X = X.reshape(X.flatten().shape[0], 1)

                # step 1
                nets, actuals = self._feedforward(X)

                # step 2
                delta_weights, delta_biases = self._backpropagation(Y, nets, actuals)

                # step 3
                self._update_weight_bias(learning_rate, delta_weights, delta_biases)

                # calculate loss in every iteration
                epoch_loss.append(np.linalg.norm(actuals[-1] - Y))

            # epoch loss is average of all losses in epoch.
            avg = np.average(epoch_loss)
            loss.append(avg)

            if e % 10 == 0: print(f'epoch {e} - loss = {avg}')

        print(f'final loss : {loss[-1]}')
        return loss

    def _feedforward(self, x):
        """
        Calculate the output of neural network

        :param x: input data to the network

        :return nets: the list of net outputs of neurons in every layer
        :return actuals: the list of actual outputs of neurons in every layer after applying activations.
        """

        a = x
        nets = []
        actuals = [a]

        for i in range(len(self.weights)):
            n = self.weights[i].dot(a) + self.biases[i]
            nets.append(n)

            a = self.activation_function(self.activations[i], n)
            actuals.append(a)

        return nets, actuals

    def _backpropagation(self, desireds, nets, actuals):
        """
        Backpropagating errors to the network

        :param desireds: desired output known as labels.
        :param nets: the list of net outputs of neurons in every layer
        :param actuals: the list of actual outputs of neurons in every layer after applying activations.

        :return delta_weights: used for update weights
        :return delta_biases: used for update biases

        """

        deltas = [0] * len(self.weights)

        # calculate the last layer error (delta)
        deltas[-1] = -2 * self.derivative_activation_function(self.activations[-1], nets[-1]) * (desireds - actuals[-1])

        # BackPropagation, calculate the delta for every layer
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.derivative_activation_function(self.activations[i], nets[i]) * self.weights[i + 1].T.dot(deltas[i + 1])

        delta_weights = [d.dot(actuals[i].T) for i, d in enumerate(deltas)]
        delta_biases = deltas

        return delta_weights, delta_biases

    def _update_weight_bias(self, learning_rate, delta_weights, delta_biases):
        """
        Update weights and biases

        :param learning_rate: the learning rate
        :param delta_weights: used for update weights
        :param delta_biases: used for update biases
        """

        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, delta_weights)]
        self.biases = [w - learning_rate * db for w, db in zip(self.biases, delta_biases)]

    def activation_function(self, name, value):
        """
        Apply activation functions

        :param name: name of function
        :param value: value

        :return new value after applying activation
        """

        if name == 'sig':
            return self.sig(value)

        elif name == 'tanh':
            return self.tanh(value)

        elif name == 'linear':
            return self.linear(value)

        elif name == 'relu':
            return self.relu(value)

        else:
            # unknown activation function
            return self.linear(value)

    def derivative_activation_function(self, name, value):
        """
        Apply derivative of activation functions

        :param name: name of function
        :param value: value

        :return new value after applying derivative of activation
        """

        if name == 'sig':
            d_sig = self.sig(value) * (1 - self.sig(value))
            return d_sig

        elif name == 'tanh':
            d_tanh = 1 - np.power(self.tanh(value), 2)
            return d_tanh

        elif name == 'linear':
            return 1

        elif name == 'relu':
            if value >= 0:
                return 1

            return 0

        else:
            # unknown
            return 1

    def predict(self, x):
        """
        Calculate the output of neural network, just like feedforward with two differences: 
            1. here we need to preprocess input 
            2. we just return a final output of network

        :param x: input data

        :return a: output of network
        """

        a = np.array(x)
        a = a.reshape(a.flatten().shape[0], 1)

        for i in range(len(self.weights)):
            n = self.weights[i].dot(a) + self.biases[i]
            a = self.activation_function(self.activations[i], n)

        return a

# Example 1 : Hex Detection

X = [
    [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1],
    ],
    [
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    [
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ],
    [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ],
]

Y = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]

for i in range(16):
    plot.imshow(X[i])
    plot.show()

epochs = 1200
nn = NeuralNetwork(structure=[64, 16, 16, 4], activations=['sig', 'sig', 'sig'])

loss = nn.learn(X, Y, epochs=epochs, learning_rate=0.05)

plt.scatter([i for i in range(0, epochs)], loss)
plt.show()

xx = [
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 1],
]

plot.imshow(xx) 
plot.show()

yy = [1, 0, 1, 0]
a = nn.predict(xx)
print(a)
print(np.around(a, 1))

xx = [
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
]

plot.imshow(xx)
plot.show()

yy = [0, 0, 0, 1]
a = nn.predict(xx)
print(a)
print(np.around(a, 1))

xx = [
    [1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
]

plot.imshow(xx)
plot.show()

yy = [1, 1, 1, 1]
a = nn.predict(xx)
print(a)
print(np.around(a, 1))

xx = [
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1],
]

plot.imshow(xx)
plot.show()

yy = [1, 0, 0, 0]
a = nn.predict(xx)
print(a)
print(np.around(a, 1))

# Example 2: Cos(x)

x = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(10000,))
y = np.cos(x)
x = np.reshape(x, (-1, 1)).tolist()
y = np.reshape(y, (-1, 1)).tolist()

epochs = 50
nn = NeuralNetwork(structure=[1, 100, 100, 1], activations=['sig', 'sig', 'tanh'])

loss = nn.learn(x, y, epochs=epochs, learning_rate=0.05)

plt.scatter([i for i in range(0, epochs)], loss)
plt.show()

x = np.random.uniform(low=0, high=2 * np.pi, size=(1000,))
y = np.cos(x)
x = np.reshape(x, (-1, 1)).tolist()
y = np.reshape(y, (-1, 1)).tolist()
outs = []
for xx in x:
    outs.append(nn.predict(xx))

plt.scatter(x, y)
plt.scatter(x, outs)
plt.show()