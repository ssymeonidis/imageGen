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
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# create tensorflow model
def gen(size, latent_dim):
  s = layers.Input((1, latent_dim))
  x = layers.Dense(256)(s)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dense(1024)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dense(4096)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dense(size[0] * size[1] * size[2])(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Reshape(size)(x)
  x = layers.Activation("tanh")(x)
  return Model(s, x, name="generator") 
