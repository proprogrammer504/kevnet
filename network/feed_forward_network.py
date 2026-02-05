from utils import Utils
from network_components import Layer


class ANN:
    def __init__(self, input_dim, layer_dim, layers, output_dim):
        """
        feature size defines input dim,
        layer size defines the number of neurons per layer,
        layers define number of hidden layers,
        output defines number of outputs
        """

        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.layers = layers
        self.output_dim = output_dim

        self.learning_rate = 1e-5
        self.network = []
        self.init_network()
    
    def forward(self, inputs):
        """
        Forward pass. Each layer takes the last layers output vector outputs a new output vector
        of dimension input dim where each entry has the transformation of the sum of all previous
        inputs multiplied by a weight + a bias.
        """
        outputs = []
        for index, layer in enumerate(self.network):
            if index == 0:
                outputs = layer.forward(inputs)
            else:
                outputs = layer.forward[outputs]
            
        return Utils.softmax(outputs)

    def backward(self):
        pass

    def init_network(self):
        """
        Initializes input hidden and output layers
        """
        for _ in range(self.layers):
            self.network.append(Layer(self.layer_dim, self.input_dim))
            self.network.append(Layer(self.output_dim), self.input_dim)