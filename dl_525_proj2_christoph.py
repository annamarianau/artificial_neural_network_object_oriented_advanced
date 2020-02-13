"""
COSC 525 - Deep Learning
Project 2
Contributors:
Metzner, Christoph
Nau, Anna-Maria
Date: 02/10/2020
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

        if weights is None:
            # wieso hast du hier 1 für weights? fully connected layer
            self.weights = np.random.uniform(0, 1, number_input)
        else:
            self.weights = weights
        # stores output computed within the feed-forward algorithm
        self.output = None
        # stores delta computed within the back-propagation algorithm
        self.delta = None
        # computed updated weights are temporarily stored in an array
        # necessary since back-propagation uses the current weights of neurons in previous layer
        self.updated_weights = []

    # Method for activation of neuron using variable z as input
    # z = bias + sum(weights*inputs)
    # If-statement to select correct activation function based on given string-input ("logistic" or "linear")
    def activate(self, z):
        if self.activation_function == "logistic":
            return log_act(z)
        elif self.activation_function == "linear":
            return lin_act(z)

    # Method for calculating output of neuron based on weighted sum
    # Computing output for neurons of ConvolutionalLayer object requires summation of all elementwise products
    def calculate_conv2d(self, input_vector):
        return self.activate(self.bias + np.sum(np.multiply(self.weights, input_vector)))

    # Computing output for neurons of a FullyConnectedLayer object requires np.dot to generate correct results
    def calculate_fully(self, input_vector):
        return self.activate(self.bias + np.dot(self.weights, input_vector))

    # Method to calculate the delta values for the neuron if in the output layer
    def calculate_delta_output(self, actual_output_network):
        if self.activation_function == "logistic":
            return -(actual_output_network - self.output[0]) * log_act_prime(self.output[0])
        elif self.activation_function == "linear":
            return -(actual_output_network - self.output[0]) * lin_act_prime(self.output[0])

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
            neuron.output = neuron.calculate_fully(input_vector=input_vector)#neuron.calculate(input_vector=input_vector)
            output_curr_layer_neuron.append(neuron.output)
        return output_curr_layer_neuron

    # this function calls neuron object method update_weights_bias() to start updating weights for next feed-forward
    # algorithm (For this ANN after each sample --> online processing)
    def update_weights_bias(self):
        for neuron in self.neurons:
            neuron.update_weights_bias()

    def backprop(self, layer_index, input_vector, actual_output, deltas_weights_previous_layer, input_network=None):
        # reverse the input_vector; actually contains outputs of all layers from feedforward algorithm
        # j = 0 --> output layer
        # j > 0 --> any hidden layer
        if layer_index == 0:
            # Creating a list which contains all deltas of all output neurons
            deltas_weights_output_layer = []
            # Loop: Compute the delta for each neuron in output_neuron
            # actual_output_network[neuron_index]: index of actual output at output neurons of network
            for neuron_index, neuron in enumerate(self.neurons):
                neuron.delta = neuron.calculate_delta_output(actual_output_network=actual_output[neuron_index])
                deltas_weights_output_layer.append((neuron.delta, neuron.weights))
                # IF-statement used to select algorithm for AND / XOR case with one perceptron
                for index_input, input in enumerate(input_vector[layer_index+1]):
                    error_weight = neuron.delta * input
                    updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                    neuron.updated_weights.append(updated_weight)
                    neuron.updated_bias = neuron.bias - self.learning_rate * neuron.delta
                print("Current weights: {} --> updated weights: {}".format(neuron.weights, neuron.updated_weights))
                print("Current bias: {} --> updated bias: {}".format(neuron.bias, neuron.updated_bias))
            return deltas_weights_output_layer
        elif layer_index > 0:
            # Loop: Compute the sum of deltas for each neuron in current layer
            for neuron_index, neuron in enumerate(self.neurons):
                # print("Updated Weights and bias for neuron {} in hidden layer {} seen from output layer:".format(neuron_index + 1, index_layer))
                # setting sum of deltas to 0 for each new neuron
                delta_sum = 0
                # computing delta_sum based on the delta values from neuron in previous layer and the weights
                # connected between those neurons and the neuron in current layer (currently in loop)
                for delta_weight in deltas_weights_previous_layer:
                    delta_sum += delta_weight[0] * delta_weight[1][neuron_index]
                neuron.delta = neuron.calculate_delta_hidden(delta_sum=delta_sum)
                if layer_index == (len(input_vector)-1):
                    for index_input, input in enumerate(input_network):
                        error_weight = neuron.delta * input
                        updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                        neuron.updated_weights.append(updated_weight)
                else:
                    for index_input, input in enumerate(input_vector[layer_index + 1]):
                        error_weight = neuron.delta * input
                        updated_weight = neuron.weights[index_input] - self.learning_rate * error_weight
                        neuron.updated_weights.append(updated_weight)
                # updating Bias of neuron
                neuron.updated_bias = neuron.bias - self.learning_rate * neuron.delta
                print("Current weights: {} --> updated weights: {}".format(neuron.weights, neuron.updated_weights))
                print("Current bias: {} --> updated bias: {}".format(neuron.bias, neuron.updated_bias))


class NeuralNetwork:
    def __init__(self, input_size_nn, learning_rate, loss_function):
        self.input_size_nn = input_size_nn
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.NetworkLayers = []
        self.output_each_layer = []
        self.current_deltas_weights = None

    def addLayer(self, layer_object):
        self.NetworkLayers.append(layer_object)

    def print_network(self):
        print(self.NetworkLayers)
        for layer in self.NetworkLayers:
            print(layer)
            print(layer.neurons)
            for neuron in layer.neurons:
                print(neuron.weights)
                print(neuron.bias)

    # Method to compute the losses at each output neuron
    # mse_loss: Mean Squared Error
    # bin_cross_entropy_loss: Binary Cross Entropy
    # predicted_output: Output after activation for each output neuron
    # actual_output: Actual output of
    def calculateloss(self, predicted_output, actual_output, number_samples=None):
        if self.loss_function == "mse":
            return mse_loss(predicted_output=predicted_output,
                            actual_output=actual_output,
                            number_samples=number_samples)

        elif self.loss_function == "bincrossentropy":
            return bin_cross_entropy_loss(predicted_output=predicted_output,
                                          actual_output=actual_output,
                                          number_samples=number_samples)

    def update_weights_bias(self):
        # reverse the order of the list containing the individual layer objects
        # necessary for next samples training iteration --> correct feed forward information
        self.FullyConnectedLayers.reverse()
        for layer in self.FullyConnectedLayers:
            layer.update_weights_bias()

    def feed_forward(self, current_input_layer):
        global output_current_layer
        for i, layer in enumerate(self.NetworkLayers):
            output_current_layer = layer.calculate(input_vector=current_input_layer)
            # store output per layer in list to use for back-propagation algorithm
            self.output_each_layer.append(output_current_layer)
            current_input_layer = output_current_layer
        # Return the final layers output
        return output_current_layer

    def backpropagation(self, input_network, actual_output):
        self.NetworkLayers.reverse()
        self.output_each_layer.reverse()
        print("Hello BackProp")
        for index_layer, layer in enumerate(self.NetworkLayers):
            if index_layer == 0:
                print("Servus")
                self.current_deltas_weights = layer.backprop(layer_index=index_layer,
                                                             input_vector=self.output_each_layer,
                                                             actual_output=actual_output,
                                                             deltas_weights_previous_layer=None)
                print("HIHIHI")
            elif index_layer > 0:
                self.current_deltas_weights = layer.backprop(layer_index=index_layer,
                                                             input_vector=self.output_each_layer,
                                                             actual_output=actual_output,
                                                             deltas_weights_previous_layer=self.current_deltas_weights,
                                                             input_network=input_network)
                print("Fehler")

    def train(self, input_network, output_network, epochs=None):
        # Train network based on given argv
        for epoch in range(epochs):
            predicted_output = self.feed_forward(current_input_layer=input_network)
            loss = self.calculateloss(predicted_output=predicted_output,
                                                   actual_output=output_network,
                                                   number_samples=2)
            print("Predicted_output: ", predicted_output)
            print("Loss at output neurons: ", loss)
            print("Total Loss: ", np.sum(loss))
            self.backpropagation(input_network=input_network, actual_output=output_network)


class ConvolutionalLayer:
    def __init__(self, number_kernels, kernel_size, activation_function, dimension_input, learning_rate,
                 bias=None, weights=None, stride=None, padding=None):
        self.number_kernels = number_kernels  # scalar number of kernels in layer
        self.kernel_size = kernel_size  # vector for kernel size -> 2D, e.g.,  [3,3] = 3x3
        self.activation_function = activation_function  # activation function, e.g., logistic or linear
        self.dimension_input = dimension_input  # dimension of inpt e.g. [2,3]: width=2, height=3
        self.learning_rate = learning_rate  # learning rate is given by NeuralNetwork object

        if stride is None:
            self.stride = 1
        else:
            self.stride = stride
        self.padding = padding

        # If bias not given by user create one random bias for each kernel of ConvolutionalLayer object
        # from a uniform distribution for whole layer this initial bias value is passed on each
        # neuron in respective layer
        if bias is None:
            self.bias = np.random.uniform(0, 1, self.number_kernels)
        else:
            self.bias = bias

        # generating number of neurons
        neurons_row = round(
            ((self.dimension_input[0] - self.kernel_size) / self.stride + 1))  # number of rows with neurons
        neurons_column = round(
            ((self.dimension_input[1] - self.kernel_size) / self.stride + 1))  # number of columns with neurons
        self.number_neurons = neurons_row * neurons_column * self.number_kernels  # number of total neurons in layer

        # List holding all neurons in layer
        self.feature_maps_layer = []
        # Loop through all kernels to generate neurons
        for kernel in range(number_kernels):
            # list holding all neuron objects of one kernel
            self.feature_map = []
            # if-statement to check whether weights were given or not...
            # if not given, then generate weights for the kernel_size (quadratic) which all neurons share
            if weights is None:
                self.weights = []
                # for loop to generate weights in 2D-matrix
                for row in range(self.kernel_size):
                    self.weights.append(np.random.uniform(0, 1, self.kernel_size))
            else:
                self.weights = weights[kernel]
            # generating neurons of feature maps, with respective weights and one bias
            for row in range(neurons_row):
                self.neurons = []
                for column in range(neurons_column):
                    self.neurons.append(
                        Neuron(activation_function=self.activation_function, number_input=self.kernel_size,
                               learning_rate=self.learning_rate, weights=self.weights, bias=self.bias[kernel]))
                self.feature_map.append(self.neurons)
            self.feature_maps_layer.append(self.feature_map)

    def calculate(self, input_vector):
        output_feature_maps = []
        for feature_map in self.feature_maps_layer:
            kern_edge_top = 0
            kern_edge_bot = self.kernel_size
            output_feature_map = []
            for neuron_row in feature_map:
                kern_edge_l = 0
                kern_edge_r = self.kernel_size
                output_neurons_row = []
                for neuron in neuron_row:
                    # generating a snippet of input matrix based on size of kernel and location
                    reshape_input_vector = input_vector[kern_edge_top:kern_edge_bot, kern_edge_l:kern_edge_r]
                    neuron.output = neuron.calculate_conv2d(input_vector=reshape_input_vector)
                    # Updating the location of kernel for next neuron via stride
                    kern_edge_l += self.stride
                    kern_edge_r += self.stride
                    output_neurons_row.append(neuron.output)
                output_feature_map.append(output_neurons_row)
                kern_edge_top += self.stride
                kern_edge_bot += self.stride
            output_feature_maps.append(output_feature_map)
        return output_feature_maps

    def backprop(self, layer_index, input_vector, actual_output, deltas_weights_previous_layer=None):
        pass


class MaxPoolingLayer:
    def __init__(self, kernel_size, dimension_input):
        self.kernel_size = kernel_size
        self.dimension_input = dimension_input
        self.stride = kernel_size
        self.maxpool_feature_maps = []
        self.maxpool_index_maps = []

    def calculate(self, input_vector):
        number_input_kernels = len(input_vector)
        number_strides_rows = round(self.dimension_input[0] / self.stride)
        number_strides_cols = round(self.dimension_input[1] / self.stride)
        for kernel_index in range(number_input_kernels):
            kern_edge_top = 0
            kern_edge_bot = self.kernel_size
            maxpool_feature_map = []
            maxpool_index_map = []
            for stride_index_row in range(number_strides_rows):
                kern_edge_l = 0
                kern_edge_r = self.kernel_size
                maxpool_row = []
                for stride_index_col in range(number_strides_cols):
                    maxpool = np.max(input_vector[kernel_index][kern_edge_top:kern_edge_bot, kern_edge_l:kern_edge_r])
                    # print("Matrix: ", input_vector[kernel_index])
                    # print("Matrix snippet: ", input_vector[kernel_index][kern_edge_top:kern_edge_bot, kern_edge_l:kern_edge_r])
                    # print("Maxpool value :", maxpool)
                    maxpool_index = np.where(input_vector[kernel_index] == maxpool)
                    # print("Maxpool index: ", maxpool_index)
                    maxpool_index_map.append((maxpool, maxpool_index))
                    maxpool_row.append(maxpool)
                    kern_edge_l += self.stride
                    kern_edge_r += self.stride
                maxpool_feature_map.append(maxpool_row)
                kern_edge_top += self.stride
                kern_edge_bot += self.stride
            self.maxpool_index_maps.append(maxpool_index_map)
            self.maxpool_feature_maps.append(maxpool_feature_map)
            return self.maxpool_feature_maps

    def backprop(self, layer_index, input_vector, actual_output, deltas_weights_previous_layer=None):
        maxpool_backprop_map = np.zeros(self.dimension_input)
        # maxpool_mask_map[maxpool_index] = maxpool
        pass


class FlattenLayer:
    def __init__(self, dimension_input):
        self.dimension_input = dimension_input

    def calculate(self, input_vector):
        return np.array(input_vector).flatten()

    def backprop(self, layer_index, input_vector, actual_output, deltas_weights_previous_layer=None):
        input_vector.reverse()
        if layer_index == 0:
            return input_vector.reshape(dimension_input).tolist()


"""
Activation Functions with their respective prime functions
- logistic (log_act)
- linear (lin_act)
- ReLU (ReLU_act)
- z: result of the weighted sum of weights (w), biases (b), and inputs (x) - z = np.dot(w,x)-b
"""


# sigmoid/logistic activation function
def log_act(z):
    return 1 / (1 + np.exp(-z))


def log_act_prime(output):
    return output * (1 - output)


# linear activation function
def lin_act(z):
    return z


def lin_act_prime(z):
    return 1


# rectified linear activation function
def ReLU_act(z):
    # return 0 if x<= 0 or x if x>0
    return np.maximum(0, z)


def ReLU_act_prime(z):
    if z <= 0:
        return 0
    elif z > 0:
        return 1


"""
Loss Functions
- Mean squared error (mse_loss); https://en.wikipedia.org/wiki/Mean_squared_error
- Binary cross entropy loss; https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html (bin_cross_entropy_loss)
- predicted: Array containing all predicted/computed output for each sample by the neural network.
- actual: Array containing the ground truth value for each sample, respectively.
"""


def mse_loss(predicted_output, actual_output, number_samples):
    output_network = []
    for output_index, output in enumerate(predicted_output):
        loss = 1 / number_samples * (actual_output[output_index] - output)**2
        output_network.append(loss)
    return output_network


def bin_cross_entropy_loss(predicted_output, actual_output, number_samples):
    return 1 / number_samples * -(actual_output * np.log(predicted_output)
                               + (1 - actual_output) * np.log(1 - predicted_output))


# Driver code main()
def main(argv=None):
    # First List: List holding all weights for each kernel
    # Second List: Holds weights for one kernel
    # Third List: Holds weights for first row of kernel weight 00, 01, 02 / 10, 11, 12 / 20, 21, 22
    # print(NN.NetworkLayers[layer_index].neurons_layer[kernel_index_neurons][row_index_neurons][column_index_neurons])
    # cov_res = [4,3,4], [2,4,3], [2, 3, 4]
    if argv[1] == 'example1':
        input_example1 = [[1, 1, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 0],
                          [0, 1, 1, 0, 0]]
        weights_example1 = [[[-0.5321357, -0.12474307, 0.35251492],
                             [-0.5045868, 0.38132334, 0.52816117],
                             [0.57359624, 0.49421835, 0.02058202]]]

        output_example1 = 0.5

        NN = NeuralNetwork(input_size_nn=2, learning_rate=0.1, loss_function="mse")
        NN.addLayer(ConvolutionalLayer(number_kernels=1, kernel_size=3, activation_function="logistic",
                                       dimension_input=[5, 5], learning_rate=0.1, bias=[0], weights=weights_example1,
                                       stride=None, padding=None))
        NN.addLayer(FlattenLayer(dimension_input=[3, 3]))
        NN.addLayer(FullyConnectedLayer(number_neurons=1, activation_function="logistic",
                                        number_input=np.product([3, 3]), learning_rate=0.1, weights=None, bias=1))
        NN.train(input_network=np.array(input_example1),
                 output_network=[output_example1],
                 epochs=1)

    elif argv[1] == 'example2':
        input_example2 = [[1, 1, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 0],
                          [0, 1, 1, 0, 0]]
        weights_example2_conv1 = [[[0.1, 0.15, 0.20], [0.25, 0.30, 0.35], [0.40, 0.45, 0.5]]]
        weights_example2_conv2 = [[[0.55, 0.6, 0.65], [0.70, 0.75, 0.80], [0.85, 0.90, 0.95]]]
        output_example2 = 0.5

        NN = NeuralNetwork(input_size_nn=2, learning_rate=0.1, loss_function="mse")
        NN.addLayer(ConvolutionalLayer(number_kernels=1, kernel_size=3, activation_function="logistic",
                                       dimension_input=[5, 5], learning_rate=0.1, bias=[2],
                                       weights=weights_example2_conv1, stride=None, padding=None))
        NN.addLayer(ConvolutionalLayer(number_kernels=1, kernel_size=3, activation_function="logistic",
                                       dimension_input=[3, 3], learning_rate=0.1, bias=[2],
                                       weights=weights_example2_conv2, stride=None, padding=None))
        NN.addLayer(FlattenLayer(dimension_input=[3, 3]))
        NN.addLayer(FullyConnectedLayer(number_neurons=1, activation_function="logistic",
                                        number_input=np.product([3, 3]), learning_rate=0.1, weights=None, bias=1))
        NN.train(input_network=np.array(input_example2),
                 output_network=[output_example2],
                 epochs=1)

    elif argv[1] == 'example3':
        input_example3 = [[1, 1, 1, 0, 1, 1],
                          [0, 1, 1, 1, 0, 1],
                          [0, 0, 1, 1, 1, 0],
                          [0, 1, 1, 0, 1, 1],
                          [0, 1, 0, 0, 1, 0],
                          [1, 1, 1, 0, 0, 1]]
        weights_example3 = [[[0.1, 0.15, 0.20], [0.25, 0.30, 0.35], [0.40, 0.45, 0.5]]]
        output_example3 = 0.5
        NN = NeuralNetwork(input_size_nn=2, learning_rate=0.1, loss_function="mse")
        NN.addLayer(ConvolutionalLayer(number_kernels=1, kernel_size=3, activation_function="logistic",
                                       dimension_input=[6, 6], learning_rate=0.1, bias=[2], weights=weights_example3,
                                       stride=None, padding=None))
        NN.addLayer(MaxPoolingLayer(kernel_size=2, dimension_input=[4, 4]))
        NN.addLayer(FlattenLayer(dimension_input=[2, 2]))
        NN.addLayer(FullyConnectedLayer(number_neurons=1, activation_function="logistic",
                                        number_input=np.product([2, 2]), learning_rate=0.1, weights=None, bias=1))
        NN.train(input_network=np.array(input_example3),
                 output_network=[output_example3],
                 epochs=1)

    elif argv[1] == "project1":
        input_network = [0.05, 0.10]
        weights_hidden = [[0.15, 0.20], [0.25, 0.30]]
        bias_hidden = [0.35]
        weights_output = [[0.40, 0.45], [0.50, 0.55]]
        bias_output = [0.60]

        output_network = [0.01, 0.99]

        NN = NeuralNetwork(input_size_nn=[2, 1], learning_rate=0.5, loss_function="mse")
        # Hidden_layer
        NN.addLayer(FullyConnectedLayer(number_neurons=2, activation_function="logistic",
                                        number_input=np.product([2, 1]), learning_rate=0.5,
                                        weights=weights_hidden, bias=bias_hidden))
        # Output_layer
        NN.addLayer(FullyConnectedLayer(number_neurons=2, activation_function="logistic",
                                        number_input=np.product([2, 1]), learning_rate=0.5,
                                        weights=weights_output, bias=bias_output))
        # NN.print_network()
        # starting the training algorithm
        NN.train(input_network=input_network, output_network=output_network, epochs=1)


if __name__ == '__main__':
    main(sys.argv)
