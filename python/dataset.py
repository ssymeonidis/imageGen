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
import os
from glob import glob

# internal constants
images_path  = "/home/simeon/Data/anime/data/*.jpg"
batch_size   = 32

# load image
def load_image(filename):
  img = tf.io.read_file(filename)
  img = tf.io.decode_jpeg(img)
  img = tf.image.resize(img, [64, 64])
  img = tf.cast(img, tf.float32)
  img = (img - 127.5) / 127.5
  return img

# create dataset
def gen():
  images_files = glob(images_path)
  print(f"images: {len(images_files)}")
  ds = tf.data.Dataset.from_tensor_slices(images_files)
  ds = ds.shuffle(buffer_size=1000).map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds
