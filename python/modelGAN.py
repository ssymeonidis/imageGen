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

# training step
@tf.function
def train_step(real_images, latent_dim, generator, discriminator, g_opt, d_opt):

  batch_size = tf.shape(real_images)[0]
  bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1) 
  noise = tf.random.normal([batch_size, 1, latent_dim])

  # update discriminator
  for _ in range(2):
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
  
  return d_loss, g_loss

