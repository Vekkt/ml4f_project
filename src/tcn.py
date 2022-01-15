from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Layer, Conv1D, ReLU, PReLU, Add, Conv2D
from tensorflow.keras.activations import relu
from tensorflow.keras.constraints import max_norm
import numpy as np


class TemporalBlock(Layer):
    def __init__(self, input_size, hidden_size, output_size, dilation, kernel_size, clip_weights=np.inf, skip=False):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1D(hidden_size, kernel_size,
                            dilation_rate=dilation,
                            padding='causal',
                            kernel_constraint=max_norm(clip_weights),
                            input_shape=(None, input_size))

        self.prel1 = PReLU(shared_axes=[1, 2])
        self.conv2 = Conv1D(output_size, kernel_size,
                            dilation_rate=dilation,
                            padding='causal',
                            kernel_constraint=max_norm(clip_weights))

        self.prel2 = PReLU(shared_axes=[1, 2])

        if input_size != output_size:
          self.downsample = Conv1D(output_size, 1, input_shape=(None, input_size), kernel_constraint=max_norm(clip_weights))
        else:
          self.downsample = None

        self.skip = skip

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.prel1(x)
        x = self.conv2(x)
        x = self.prel2(x)

        if self.skip:
          skip = x if self.downsample is None else self.downsample(x)
          return x + skip
        else:
          return x


class TCN(Model):
    def __init__(self, input_size, hidden_size, output_size, clip_weights=np.inf, tcn_skip=True, block_skip=False):
        super(TCN, self).__init__()

        self.tcn_skip = tcn_skip
        self.block_skip = block_skip

        self.modules = [TemporalBlock(
            input_size, hidden_size, hidden_size, 1, 1, clip_weights, block_skip)]

        for i in range(6):
            self.modules.append(TemporalBlock(
                hidden_size, hidden_size, hidden_size, 2, 2**i, clip_weights, block_skip))

        self.skip = Add()
        self.conv = Conv1D(output_size, 1, dilation_rate=1, kernel_constraint=max_norm(clip_weights))

    def call(self, inputs):
        out = inputs
        out_layers = []
        for temporalBlock in self.modules:
            out = temporalBlock(out)
            out_layers.append(out)

        if self.tcn_skip:
          out = self.skip(out_layers)
        out = self.conv(out)
        return out

    def model(self, input_shape):
        x = Input(shape=input_shape)
        return Model(inputs=[x], outputs=self.call(x))
