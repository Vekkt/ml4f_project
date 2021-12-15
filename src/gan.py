from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import sigmoid
from tcn import TCN


class Generator(Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.tcn = TCN(input_size, hidden_size, output_size)
        
    def call(self, inputs):
        return self.tcn(inputs)
    
    
class Discriminator(Layer):
    def __init__(self,  input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.tcn = TCN(input_size, hidden_size, output_size)
        
    def call(self, inputs):
        return sigmoid(self.tcn(inputs))
        