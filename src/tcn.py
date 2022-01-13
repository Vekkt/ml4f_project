from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Layer, Conv1D, ReLU, PReLU, Add, Conv2D
from tensorflow.keras.activations import relu


class TemporalBlock(Layer):
    def __init__(self, input_size, hidden_size, output_size, dilation, kernel_size):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1D(hidden_size, kernel_size,
                            dilation_rate=dilation,
                            padding='causal',
                            input_shape=(None, input_size))
        self.prel1 = PReLU(shared_axes=[1, 2])
        self.conv2 = Conv1D(output_size, kernel_size,
                            dilation_rate=dilation,
                            padding='causal')
        self.prel2 = PReLU(shared_axes=[1, 2])

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.prel1(x)
        x = self.conv2(x)
        x = self.prel2(x)
        return x


class TCN(Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(TCN, self).__init__()
        self.modules = [TemporalBlock(
            input_size, hidden_size, hidden_size, 1, 1)]

        for i in range(6):
            self.modules.append(TemporalBlock(
                hidden_size, hidden_size, hidden_size, 2, 2**i))

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

    def model(self, input_shape):
        x = Input(shape=input_shape)
        return Model(inputs=[x], outputs=self.call(x))
