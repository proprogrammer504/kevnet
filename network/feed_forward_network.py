class ANN:
    def __init__(self, feature_size, layer_size, layers, output_size):
        """
        feature size defines input dim,
        layer size defines the number of neurons per layer,
        layers define number of hidden layers,
        output defines number of outputs
        """

        self.feature_size = feature_size
        self.layer_size = layer_size
        self.layers = layers
        self.output = output_size
        
        self.bias = 0
        self.weights = self.init_network()
        self.learning_rate = 1e-5
    
    def forward(self):
        pass

    def backward(self):
        pass

    def init_network(self):
        weights = []

        for _ in range(0, self.layers):
            weights.append([0] * self.layer_size)

        return weights