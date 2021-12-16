import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.activations import sigmoid
from tcn import TCN


class Generator(Layer):
    def __init__(self, input_size, hidden_size, output_size, rfs=127):
        super(Generator, self).__init__()
        self.tcn = TCN(input_size, hidden_size, output_size, rfs)

    def call(self, inputs):
        return self.tcn(inputs)


class Discriminator(Layer):
    def __init__(self,  input_size, hidden_size, output_size, rfs=127):
        super(Discriminator, self).__init__()
        self.tcn = TCN(input_size, hidden_size, output_size, rfs)

    def call(self, inputs):
        return sigmoid(self.tcn(inputs))


class GAN(Model):
    def __init__(self, latent_size, hidden_size, output_size, d_train_steps=5):
        super(GAN, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_train_steps = d_train_steps
        self.generator = Generator(latent_size, hidden_size, output_size)
        self.discriminator = Discriminator(
            output_size, hidden_size, output_size)

    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = Mean(name="d_loss")
        self.g_loss_metric = Mean(name="g_loss")

    def loss_fn(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.log(y_true)) + tf.reduce_mean(tf.math.log(1. - y_pred))

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        rfs = tf.shape(real_data)[1]

        # Train the discriminator
        for _ in range(self.d_train_steps):
            latent_noise = tf.random.normal(shape=(batch_size, rfs, self.latent_size))
            fake_data = self.generator(latent_noise)

            with tf.GradientTape() as tape:
                pred_fake = self.discriminator(fake_data)
                pred_real = self.discriminator(real_data)
                d_loss = self.loss_fn(pred_real, pred_fake)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train the generator
        latent_noise = tf.random.normal(shape=(batch_size, rfs, self.latent_size))
        
        with tf.GradientTape() as tape:
            pred_fake = self.discriminator(self.generator(latent_noise))
            pred_misleading = tf.zeros((batch_size, rfs, self.output_size))
            g_loss = self.loss_fn(pred_fake, pred_misleading)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update the parameters
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
