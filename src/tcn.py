import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class TemporalBlockModule(layers.Layer):
    def __init__(self, input_size, hidden_size, output_size, dilation, kernel_size):
        super(TemporalBlockModule, self).__init__()
        self.block = Sequential()
        self.block.add(layers.Conv1D(hidden_size, kernel_size,
                                     dilation_rate=dilation,
                                     padding='causal',
                                     input_shape=(None, input_size)))
        self.block.add(layers.ReLU())  # NEED TO CHANGE THAT TO PRELU
        self.block.add(layers.Conv1D(output_size, kernel_size,
                                     dilation_rate=dilation,
                                     padding='causal'))
        self.block.add(layers.ReLU())  # NEED TO CHANGE THAT TO PRELU

    def call(self, inputs):
        return self.block(inputs)


class TCN(layers.Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super(TCN, self).__init__()
        self.modules = [TemporalBlockModule(
            input_size, hidden_size, hidden_size, 1, 1)]

        for i in range(6):
            self.modules.append(TemporalBlockModule(
                hidden_size, hidden_size, hidden_size, 2, 2**i))

        self.conv = layers.Conv1D(output_size, 1, dilation_rate=1)

    def call(self, inputs):
        out = inputs
        out_layers = [out]
        for temporalBlock in self.modules:
            out = temporalBlock(out)
            out_layers.append(out)
            
        out = layers.add(out_layers)
        print(out.shape)
        out = self.conv(out)
        return out
