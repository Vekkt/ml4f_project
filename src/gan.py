import numpy as np

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.activations import sigmoid
from tcn import TCN


class Generator(Layer):
    def __init__(self, input_size, hidden_size, output_size, clip_weights=np.inf, tcn_skip=True, block_skip=False):
        super(Generator, self).__init__()
        self.tcn = TCN(input_size, hidden_size, output_size, clip_weights=clip_weights, tcn_skip=tcn_skip, block_skip=block_skip)

    def call(self, inputs):
        return self.tcn(inputs)


class Discriminator(Layer):
    def __init__(self,  input_size, hidden_size, output_size, clip_weights=np.inf, tcn_skip=True, block_skip=False):
        super(Discriminator, self).__init__()
        self.tcn = TCN(input_size, hidden_size, output_size, clip_weights=clip_weights, tcn_skip=tcn_skip, block_skip=block_skip)

    def call(self, inputs):
        return sigmoid(self.tcn(inputs))


class GAN(Model):
    def __init__(self, latent_size, hidden_size, output_size, d_train_steps=5, gp_weight=10, clip_weights=np.inf, tcn_skip=True, block_skip=False):
        super(GAN, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_train_steps = d_train_steps
        self.gp_weight = gp_weight
        self.generator = Generator(latent_size, hidden_size, output_size, clip_weights, tcn_skip=tcn_skip, block_skip=block_skip)
        self.discriminator = Discriminator(
            output_size, hidden_size, output_size, clip_weights, tcn_skip=tcn_skip, block_skip=block_skip)

    def compile(self, d_optimizer, g_optimizer, loss_fn=None, use_reduce_loss=False):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.use_reduce_loss = use_reduce_loss

        if not self.use_reduce_loss:
          self.d_loss_metric = Mean(name="d_loss")
          self.g_loss_metric = Mean(name="g_loss")
          if loss_fn is None:
            raise ValueError("Loss function can't be None if use_reduce_loss is False")
          self.loss_fn = loss_fn

    def gradient_penalty(self, batch_size, real_data, fake_data):
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)

        diff = fake_data - real_data    
        
        interpolated = real_data + alpha * diff
        with tf.GradientTape() as g:
            g.watch(interpolated)
            pred = self.discriminator(interpolated)
        grads = g.gradient(pred, [interpolated])[0]
        norm = tf.linalg.norm(grads)   
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp


    def discriminator_loss(self, pred_real, pred_fake):
      if self.use_reduce_loss:
        return tf.reduce_mean(pred_fake) - tf.reduce_mean(pred_real)
      else:
        return self.loss_fn(tf.ones_like(pred_real), pred_real) + \
        self.loss_fn(tf.zeros_like(pred_fake), pred_fake)

    def generator_loss(self, pred_fake):
      if self.use_reduce_loss:
        return tf.reduce_mean(pred_fake)
      else:
        return self.loss_fn(tf.ones_like(pred_fake), pred_fake)
      
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        rfs = tf.shape(real_data)[1]

        # Train the discriminator
        for _ in range(self.d_train_steps):
            latent_noise = tf.random.normal(
                shape=(batch_size, rfs, self.latent_size))

            with tf.GradientTape() as tape:
                fake_data = self.generator(latent_noise)
                pred_fake = self.discriminator(fake_data)
                pred_real = self.discriminator(real_data)

                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                d_loss = self.discriminator_loss(pred_real, pred_fake) + self.gp_weight * gp 
            
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

        # Train the generator
        latent_noise = tf.random.normal(
            shape=(batch_size, rfs, self.latent_size))

        with tf.GradientTape() as tape:
            pred_fake = self.discriminator(self.generator(latent_noise))
            g_loss = self.generator_loss(pred_fake)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        if not self.use_reduce_loss:
          # Update the loss
          self.d_loss_metric.update_state(d_loss)
          self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": d_loss,
            "g_loss": g_loss
        }
