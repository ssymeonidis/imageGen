#################################################################
# imageGen Tensorflow Project 
# Copyright (C) 2024 Simeon Symeonidis
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#################################################################


# import libraries
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.optimizers import Adam
import modelGeneratorDense as modelGenerator
import modelDiscriminatorDense as modelDiscriminator
import plotSave

# define internal constants
n_samples = 100
d_train_iter = 2

# training step
@tf.function
def train_step(real_images, latent_dim, generator, discriminator, g_opt, d_opt):

  # initialize state
  batch_size = tf.shape(real_images)[0]
  bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1) 
  noise = tf.random.normal([batch_size, 1, latent_dim])

  # update discriminator
  for _ in range(d_train_iter):
    with tf.GradientTape() as dtape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(real_images, training=True)
      fake_output = discriminator(generated_images, training=True)
      d_real_loss = bce_loss(tf.ones_like(real_output), real_output)
      d_fake_loss = bce_loss(tf.zeros_like(fake_output), fake_output)
      d_loss = d_real_loss + d_fake_loss
      d_grad = dtape.gradient(d_loss, discriminator.trainable_variables)
      d_opt.apply_gradients(zip(d_grad, discriminator.trainable_variables))

  # update generator
  with tf.GradientTape() as gtape:
    generated_images = generator(noise, training=True)
    fake_output = discriminator(generated_images, training=True)
    g_loss = bce_loss(tf.ones_like(fake_output), fake_output)
    g_grad = gtape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grad, generator.trainable_variables))
  
  # exit function
  return d_loss, g_loss


# top-level generative adversarial networks class
class GAN:

  # constructor
  def __init__(self, size, latent_dim):
    self.g_model = modelGenerator.gen(size, latent_dim)
    self.d_model = modelDiscriminator.gen(size)
    self.latent_dim = latent_dim

  # print summary
  def summary(self):
    self.g_model.summary()
    self.d_model.summary()

  # train function
  def train(self, dataset, num_epochs, num_imgs):
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9)
    seed = np.random.normal(size=(n_samples, 1, self.latent_dim))
    
    for epoch in range(num_epochs):
      start = time.time()
      d_loss = 0.0
      g_loss = 0.0
      for image_batch in dataset:
        d_batch_loss, g_batch_loss = train_step(image_batch, self.latent_dim, self.g_model, self.d_model, g_optimizer, d_optimizer)
        d_loss += d_batch_loss
        g_loss += g_batch_loss
      d_loss = d_loss/num_imgs
      g_loss = g_loss/num_imgs

      self.g_model.save("../models/g_model.keras")
      self.d_model.save("../models/d_model.keras")

      examples = self.g_model.predict(seed, verbose=0)
      plotSave.saveNxN(examples, epoch, np.sqrt(n_samples))

      time_taken = time.time() - start
      print(f"[{epoch+1:1.0f}/{num_epochs}] {time_taken:2.2f}s - d_loss: {d_loss:1.4f} - g_loss: {g_loss:1.4f}")
