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
def gen(size):
  s = layers.Input(size)
  x = layers.Flatten()(s)
  x = layers.Dense(4096)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(1024)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(256)(x)
  x = layers.LeakyReLU(0.2)(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(1)(x)
  return Model(s, x, name="discriminator")
