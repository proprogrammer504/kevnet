import random

from utils import Utils


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def output(self, inputs):
        return Utils.relu(Utils.dot(inputs, self.weights) + self.bias)
    

class Layer:
    def __init__(self, layer_dim, input_dim):
        """
        Layer object which intakes intakes a vector and outputs a vector of same size
        """
        self.layer_dim = layer_dim
        self.input_dim = input_dim

        self.neurons = []

        self.init_layer()

    def init_layer(self):
        """
        initializes the network layer_dim number of neurons 
        """
        for _ in range(self.layer_dim):
            weights = [random.random() for _ in range(self.input_dim)]
            bias = random.random()
            self.neurons.append(Neuron(weights, bias))
    
    def forward(self, inputs):
        """
        Given an input vector, returns an output vector of same size
        """
        outputs = []

        for neuron in self.neurons:
            output = neuron.output(inputs)
            outputs.append(output)
        
        return outputs