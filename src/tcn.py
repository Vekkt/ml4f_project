    from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv1D, ReLU, Add


class TemporalBlock(Layer):
    def __init__(self, input_size, hidden_size, output_size, dilation, kernel_size, rfs=127):
        super(TemporalBlock, self).__init__()
        self.block = Sequential()
        self.block.add(Conv1D(hidden_size, kernel_size,
                                     dilation_rate=dilation,
                                     padding='causal',
                                     input_shape=(None, input_size)))
        self.block.add(ReLU())
        self.block.add(Conv1D(output_size, kernel_size,
                                     dilation_rate=dilation,
                                     padding='causal'))
        self.block.add(ReLU())

    def call(self, inputs):
        return self.block(inputs)


class TCN(Layer):
    def __init__(self, input_size, hidden_size, output_size, rfs=127):
        super(TCN, self).__init__()
        self.modules = [TemporalBlock(
            input_size, hidden_size, hidden_size, 1, 1, rfs)]

        for i in range(6):
            self.modules.append(TemporalBlock(
                hidden_size, hidden_size, hidden_size, 2, 2**i, rfs))

        self.skip = Add()
        self.conv = Conv1D(output_size, 1, dilation_rate=1)

    def call(self, inputs):
        out = inputs
        out_layers = []
        for temporalBlock in self.modules:
            out = temporalBlock(out)
            out_layers.append(out)

        out = self.skip(out_layers)
        out = self.conv(out)
        return out
