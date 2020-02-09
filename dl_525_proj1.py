"""
COSC 525 - Deep Learning
Project 1
Contributors:
Metzner, Christoph
Nau, Anna-Maria
Date: 01/24/2020
"""

# Imported Libraries
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt


"""
This program describes an artificial neural network (ANN) developed with object-oriented programming using Python 3.
An ANN will consist out of the following three classes:
- Neuron
- FullyConnectedLayer
- NeuralNetwork
Each class represents on its own a distinct level of scale for a typical ANN.
"""


class Neuron:
    def __init__(self, activation_function, number_input, learning_rate, weights=None, bias=None):
        self.activation_function = activation_function
        self.number_input = number_input
        self.learning_rate = learning_rate
        self.bias = bias
        self.updated_bias = None

        # stores output computed within the feed-forward algorithm
        self.output = None
        # stores delta computed within the back-propagation algorithm
        self.delta = None
        # computed updated weights are temporarily stored in an array
        # necessary since back-propagation uses the current weights of neurons in previous layer
        self.updated_weights = []

        # If-statement to check if user gave weights or not
        # if no weights given, then generate weights from a uniform distribution
        if weights is None:
            self.weights = np.random.uniform(0, 1, self.number_input)
        else:
            self.weights = weights

    # Method for activation of neuron using variable z as input
    # z = bias + sum(weights*inputs)
    # If-statement to select correct activation function based on given string-input ("logistic" or "linear")
    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    # Method for calculating output of neuron based on weighted sum
    def calculate(self, input_vector):
        return self.activate(self.bias + np.dot(self.weights, input_vector))

    # Method to calculate the delta values for the neuron if in the output layer
    def calculate_delta_output(self, actual_output_network):
        if self.activation_function == "logistic":
            return -(actual_output_network - self.output) * log_act_prime(self.output)
        elif self.activation_function == "linear":
            return -(actual_output_network - self.output) * lin_act_prime(self.output)

    # Method to calculate the delta values for the neuron if in the hidden layer
    def calculate_delta_hidden(self, delta_sum):
        if self.activation_function == "logistic":
            return delta_sum * log_act_prime(self.output)
        elif self.activation_function == "linear":
            return delta_sum * lin_act_prime(self.output)

    # Method which updates the weight and bias
    # class attributes self.updated_weights and self.updated_bias are cleared or set to None
    # for next training sample iteration
    def update_weights_bias(self):
        self.weights = copy.deepcopy(self.updated_weights)
        self.updated_weights.clear()
        self.bias = copy.deepcopy(self.updated_bias)
        self.updated_bias = None


class FullyConnectedLayer:
    def __init__(self, number_neurons, activation_function, number_input, learning_rate, weights=None, bias=None):
        self.number_neurons = number_neurons
        self.activation_function = activation_function
        self.number_input = number_input
        self.learning_rate = learning_rate

        # If bias not given by user create one random bias from a uniform distribution for whole layer
        # this initial bias value is passed on each neuron in respective layer
        if bias is None:
            self.bias = np.random.uniform(0, 1)
        else:
            self.bias = bias
        # self.weights stores all weights for each neuron in the layer
        # those weights are passed down to each respective neuron
        self.weights = weights
        self.neurons = []
        # If no weights given neurons are created without weights (weights are generated in neuron object) and stored in
        # a list; otherwise weights are passed ot respective neuron
        if weights is None:
            for i in range(self.number_neurons):
                self.neurons.append(Neuron(activation_function=activation_function, number_input=self.number_input,
                                           learning_rate=self.learning_rate, weights=None, bias=self.bias))
        else:
            for i in range(self.number_neurons):
                self.neurons.append(Neuron(activation_function=activation_function, number_input=self.number_input,
                                           learning_rate=self.learning_rate, weights=self.weights[i], bias=self.bias))

    # Method calculates the output of each neuron based on the sum of the weights * input + bias of this neuron
    # storing computed output of each neuron in the neuron --> later used for back propagation
    # returns array with final output --> necessary to compute the total accrued loss
    def calculate(self, input_vector):
        output_curr_layer_neuron = []
        for neuron in self.neurons:
            neuron.output = neuron.calculate(input_vector=input_vector)
            #print("Output Neuron: ", neuron.output)
            output_curr_layer_neuron.append(neuron.output)
        return output_curr_layer_neuron

    # this function calls neuron object method update_weights_bias() to start updating weights for next feed-forward
    # algorithm (For this ANN after each sample --> online processing)
    def update_weights_bias(self):
        for neuron in self.neurons:
            neuron.update_weights_bias()


class NeuralNetwork:
    def __init__(self, number_layers, number_neurons_layer, loss_function, activation_functions_layer, number_input_nn,
                 learning_rate, weights=None, bias=None, xor_advanced=None):
        self.number_layers = number_layers  # Scalar-value (e.g., 2 - 1 hidden and 1 output layer)
        self.number_neurons_layer = number_neurons_layer  # Array (e.g., [2,2] - 2 Neurons HL and 2 Neurons in OL)
        self.loss_function = loss_function.lower()  # String-variable (e.g., "MSE" or "BCE")
        self.activation_functions_layer = activation_functions_layer  # Array with string-variables (e.g.,
        # ['logistic', 'logistic'] for two layer architecture])
        self.number_input_nn = number_input_nn  # Scalar-value (e.g., 2 - 2 "input neurons")
        self.learning_rate = learning_rate  # Scalar-value (e.g., 0.5 - passed down to each neuron)
        self.bias = bias  # Bias is generated in or passed to FullyConnectedLayer
        self.weights = weights  # Weights are passed to FullyConnectedLayer / Neuron object
        self.xor = xor_advanced

        self.FullyConnectedLayers = []
        for i in range(self.number_layers):
            if weights is None:
                # IF-statement necessary to determine the number of inputs into neurons of certain layer
                # i == 0 --> first hidden layer --> number of input in neurons determined by inputs into network
                # i != 0 --> all successive layers --> number of input in neurons determined by
                # number of neurons in previous layer
                if i == 0:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=self.number_input_nn,
                                                                         learning_rate=self.learning_rate,
                                                                         weights=None, bias=None))
                else:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=(
                                                                             self.number_neurons_layer[i - 1]),
                                                                         learning_rate=self.learning_rate,
                                                                         weights=None, bias=None))
            else:
                if i == 0:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=self.number_input_nn,
                                                                         learning_rate=self.learning_rate,
                                                                         weights=self.weights[i], bias=self.bias[i]))
                else:
                    self.FullyConnectedLayers.append(FullyConnectedLayer(number_neurons=self.number_neurons_layer[i],
                                                                         activation_function=
                                                                         self.activation_functions_layer[i],
                                                                         number_input=(
                                                                             self.number_neurons_layer[i - 1]),
                                                                         learning_rate=self.learning_rate,
                                                                         weights=self.weights[i], bias=self.bias[i]))

    # Method to compute the losses at each output neuron
    # mse_loss: Mean Squared Error
    # bin_cross_entropy_loss: Binary Cross Entropy
    # predicted_output: Output after activation for each output neuron
    # actual_output: Actual output of network
    def calculateloss(self, predicted_output, actual_output, number_samples = None):
        if self.loss_function == "mse":
            return mse_loss(predicted_output, actual_output)
        elif self.loss_function == "bincrossentropy":
            return bin_cross_entropy_loss(predicted_output, actual_output, number_samples)

    # Method for Feed-Forward algorithm
    # Computing output for each layer by calling Method (.calculate(current_input)) from FullyConnectedLayer object
    # Setting generated output of current layer equal with input_vector
    # --> Next layer / iteration uses output values as input values
    # returns computed output from neurons at final - output layer --> used to compute the loss
    def feed_forward(self, input_vector):
        global output_curr_layer
        for i, layer in enumerate(self.FullyConnectedLayers):
            #print("Layer: ", i+1)
            output_curr_layer = layer.calculate(input_vector=input_vector)
            input_vector = output_curr_layer
            # Return Output of layer
        return output_curr_layer

    # Method for Back-Propagation algorithm: Used for updating weights and biases of all neurons in all layers
    def back_propagation(self, input_vector, actual_output_network):
        # Reverse list containing layer objects since back propagation
        # starts updating with weights connected to output layer
        self.FullyConnectedLayers.reverse()
        for index_layer, layer in enumerate(self.FullyConnectedLayers):
            # j = 0 --> output layer
            # j > 0 --> any hidden layer
            if index_layer == 0:
                #print("Output Layer")
                # Loop: Compute the delta for each neuron in output_neuron
                # actual_output_network[neuron_index]: index of actual output at output neurons of network
                for neuron_index, neuron in enumerate(layer.neurons):
                    #print("Updated Weights and bias for neuron {} in output layer:".format(neuron_index+1))
                    # IF-statement used to determine number of neurons in output layer
                    if len(self.FullyConnectedLayers[0].neurons) == 1:
                        neuron.delta = neuron.calculate_delta_output(actual_output_network=actual_output_network)
                    else:
                        neuron.delta = neuron.calculate_delta_output(actual_output_network=actual_output_network[neuron_index])
                    # IF-statement used to select algorithm for AND / XOR case with one perceptron
                    if len(self.FullyConnectedLayers) == 1:
                        for index_input, input in enumerate(input_vector):
                            error_weight = neuron.delta * input
                            updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                            neuron.updated_weights.append(updated_weight)
                        #print("Current weights: {} --> updated weights: {}".format(neuron.weights, neuron.updated_weights))
                    elif len(self.FullyConnectedLayers) > 1:
                        # regular network, get outputs from neurons of following hidden layer
                        # number of outputs is equal to number of weights of neuron in current layer
                        for hid_neuron_index, hid_neuron in enumerate(
                                self.FullyConnectedLayers[index_layer + 1].neurons):
                            error_weight = neuron.delta * hid_neuron.output
                            updated_weight = neuron.weights[hid_neuron_index] - self.learning_rate * error_weight
                            neuron.updated_weights.append(updated_weight)
                    #print("Current weights: {} --> updated weights: {}".format(neuron.weights, neuron.updated_weights))
                    # updating Bias of neuron
                    neuron.updated_bias = neuron.bias - self.learning_rate * neuron.delta
                    #print("Current bias: {} --> updated bias: {}".format(neuron.bias, neuron.updated_bias))
            # Back-Propagation algorithm for hidden layers
            elif index_layer > 0:
                #print("Hidden Layer")
                # Loop: Compute the sum of deltas for each neuron in current layer
                for neuron_index, neuron in enumerate(layer.neurons):
                    #print("Updated Weights and bias for neuron {} in hidden layer {} seen from output layer:".format(neuron_index + 1, index_layer))
                    # setting sum of deltas to 0 for each new neuron
                    delta_sum = 0
                    # computing delta_sum based on the delta values from neuron in previous layer and the weights
                    # connected between those neurons and the neuron in current layer (currently in loop)
                    for previous_neuron in self.FullyConnectedLayers[index_layer-1].neurons:
                        delta_sum += previous_neuron.delta * previous_neuron.weights[neuron_index]
                    neuron.delta = neuron.calculate_delta_hidden(delta_sum=delta_sum)
                    # IF last hidden layer of network reached use original input values into network instead of
                    # the outputs from neurons of next layer
                    if index_layer == (len(self.FullyConnectedLayers)-1):
                        for index_input, input in enumerate(input_vector):
                            error_weight = neuron.delta * input
                            updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                            neuron.updated_weights.append(updated_weight)
                        #print("Current weights: {} --> updated weights: {}".format(neuron.weights, neuron.updated_weights))
                    else:
                        for hid_neuron_index, hid_neuron in enumerate(
                                self.FullyConnectedLayers[index_layer + 1].neurons):
                            error_weight = neuron.delta * hid_neuron.output
                            updated_weight = neuron.weights[hid_neuron_index] - self.learning_rate * error_weight
                            neuron.updated_weights.append(updated_weight)
                        #print("Current weights: {} --> updated weights: {}".format(neuron.weights, neuron.updated_weights))
                    # updating Bias of neuron
                    neuron.updated_bias = neuron.bias - self.learning_rate * neuron.delta
                    #print("Current bias: {} --> updated bias: {}".format(neuron.bias, neuron.updated_bias))

    def update_weights_bias(self):
        # reverse the order of the list containing the individual layer objects
        # necessary for next samples training iteration --> correct feed forward information
        self.FullyConnectedLayers.reverse()
        for layer in self.FullyConnectedLayers:
            layer.update_weights_bias()

    def train(self, input_vector, actual_output_network, argv=None, epochs=None):
        global total_loss
        # Train network based on given argv
        if argv == 'example':
            total_loss_list = []
            for i in range(epochs):
                #print("FeedForward Algorithm")
                #print("#####################")
                predicted_output_network = self.feed_forward(input_vector)
                if self.loss_function == 'mse':
                    total_loss = np.sum(self.calculateloss(predicted_output_network, actual_output_network))
                elif self.loss_function == 'bincrossentropy':
                    # predicted_output_network and actual_output_network are given as an array
                    # loss functions only works with scalar values thus... list.pop(0) to get first value of list
                    predicted_output = predicted_output_network.pop(0)
                    actual_output = actual_output_network.pop(0)
                    total_loss = np.sum(self.calculateloss(predicted_output, actual_output, len(input_vector)))
                total_loss_list.append(total_loss)
                #print("BackPropagation Algorithm")
                #print("#########################")
                self.back_propagation(input_vector, actual_output_network)
                self.update_weights_bias()
                print("Current Epoch {}".format(i+1))
                print("Input Sample: {}".format(input_vector))
                print("Final Output of Network: {}".format(predicted_output_network))
                print('Total Loss Network: {}'.format(total_loss))
            return(total_loss_list)

        elif argv == 'and':
            total_loss_all_epochs= []
            for i in range(epochs):
                total_loss_epoch = []
                for sample_index, sample in enumerate(input_vector):
                    #print("FeedForward Algorithm")
                    #print("#####################")
                    predicted_output_network = self.feed_forward(sample)
                    if self.loss_function == 'mse':
                        total_loss = np.sum(self.calculateloss(predicted_output_network, actual_output_network[sample_index]))
                    elif self.loss_function == 'bincrossentropy':
                        # predicted_output_network and actual_output_network are given as an array
                        # loss functions only works with scalar values thus... list.pop(0) to get first value of list
                        predicted_output = predicted_output_network.pop(0)
                        total_loss = np.sum(self.calculateloss(predicted_output, actual_output_network[sample_index], len(input_vector)))
                    total_loss_epoch.append(total_loss)
                    #print("BackPropagation Algorithm")
                    #print("#########################")
                    self.back_propagation(sample, actual_output_network[sample_index])
                    self.update_weights_bias()
                    if i == epochs-1:
                        print("Input Sample: {}".format(input_vector[sample_index]))
                        print("Final Output of Network: {}".format(predicted_output_network))
                        print('Total Loss Network: {}'.format(total_loss))
                total_loss_all_epochs.append(np.mean(total_loss_epoch))
            return(total_loss_all_epochs)

        elif argv == 'xor':
            total_loss_all_epochs = []
            for i in range(epochs):
                total_loss_epoch = []
                for sample_index, sample in enumerate(input_vector):
                    #print("FeedForward Algorithm")
                    #print("#####################")
                    predicted_output_network = self.feed_forward(sample)
                    if self.loss_function == 'mse':
                        total_loss = np.sum(self.calculateloss(predicted_output_network, actual_output_network[sample_index]))
                    elif self.loss_function == 'bincrossentropy':
                        # predicted_output_network and actual_output_network are given as an array
                        # loss functions only works with scalar values thus... list.pop(0) to get first value of list
                        predicted_output = predicted_output_network.pop(0)
                        total_loss = np.sum(self.calculateloss(predicted_output, actual_output_network[sample_index], len(input_vector)))
                    total_loss_epoch.append(total_loss)
                    #print("BackPropagation Algorithm")
                    #print("#########################")
                    self.back_propagation(sample, actual_output_network[sample_index])
                    self.update_weights_bias()
                    if i == epochs - 1:
                        print("Input Sample: {}".format(input_vector[sample_index]))
                        print("Final Output of Network: {}".format(predicted_output_network))
                        print('Total Loss Network: {}'.format(total_loss))
                total_loss_all_epochs.append(np.mean(total_loss_epoch))
            return(total_loss_all_epochs)


"""
Activation Functions with their respective prime functions
- logistic (log_act)
- linear (lin_act)
- z: result of the weighted sum of weights (w), biases (b), and inputs (x) - z = np.dot(w,x)-b
"""


def log_act(z):
    return 1 / (1 + np.exp(-z))


def log_act_prime(output):
    return output * (1 - output)


def lin_act(z):
    return z


def lin_act_prime(z):
    z = 1
    return z


"""
Loss Functions
- Mean squared error (mse_loss); https://en.wikipedia.org/wiki/Mean_squared_error
- Binary cross entropy loss; https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html (bin_cross_entropy_loss)
- predicted: Array containing all predicted/computed output for each sample by the neural network.
- actual: Array containing the ground truth value for each sample, respectively.
"""


def mse_loss(predicted_output, actual_output):
    return np.square(np.subtract(predicted_output, actual_output)) * 1 / 2


def bin_cross_entropy_loss(predicted_output, actual_output, num_samples):
    return 1 / num_samples * -(actual_output * np.log(predicted_output) + (1 - actual_output) * np.log(1 - predicted_output))


# Driver code main()
def main(argv=None):
    if argv[1] == 'example':
        # single step of back-propagation using example from class
        example_input = [0.05, 0.10]
        example_output = [0.01, 0.99]
        example_weights = [[(0.15, 0.20), (0.25, 0.30)], [(0.40, 0.45), (0.50, 0.55)]]
        example_biases = [0.35, 0.60]
        print("ANN with 1 Hidden Layer (2 Neurons) and Output Layer (2 Neurons) - Final Output")
        NN_example = NeuralNetwork(number_layers=2, number_neurons_layer=[2, 2], loss_function='MSE',
                                   activation_functions_layer=['logistic', 'logistic'], number_input_nn=2,
                                   learning_rate=0.5,
                                   weights=example_weights, bias=example_biases)
        NN_example.train(input_vector=example_input, actual_output_network=example_output, argv=argv[1], epochs=2)

    elif argv[1] == 'and':
        and_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
        and_output = [0, 0, 0, 1]
        print("Perceptron with Logic Gate: AND (1000 Epochs)")


        NN_and = NeuralNetwork(number_layers=1, number_neurons_layer=[1], loss_function='mse',
                           activation_functions_layer=['logistic'], number_input_nn=2, learning_rate=6,
                           weights=None, bias=None)
        NN_and.train(input_vector=and_input, actual_output_network=and_output, argv=argv[1], epochs=1000)

    elif argv[1] == 'xor':
        xor_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
        xor_output = [0, 1, 1, 0]
        print("Perceptron with Logic Gate: XOR (1000 Epochs)")
        NN_xor_normal = NeuralNetwork(number_layers=1, number_neurons_layer=[1], loss_function='MSE',
                           activation_functions_layer=['logistic'], number_input_nn=2, learning_rate=8,
                           weights=None, bias=None)
        NN_xor_normal.train(input_vector=xor_input, actual_output_network=xor_output, argv=argv[1], epochs=1000)
        print()
        print("Single-Layer NN (2 Inputs, 2 Hidden Neurons, 1 Output Neuron) for Logic Gate : XOR (10000 Epochs)")
        NN_xor_advanced = NeuralNetwork(number_layers=2, number_neurons_layer=[2, 1], loss_function='MSE',
                           activation_functions_layer=['logistic', 'logistic'], number_input_nn=2, learning_rate=8,
                           weights=None, bias=None)
        NN_xor_advanced.train(input_vector=xor_input, actual_output_network=xor_output, argv=argv[1], epochs=10000)



if __name__ == '__main__':
    main(sys.argv)
