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
import imgDataset
import modelGAN

if __name__ == '__main__':
  img_size   = (64, 64, 3)
  latent_dim = 64
  num_epochs = 1000
  dataset  = imgDataset.gen()
  num_imgs = imgDataset.num_images()
  print(f"images: {num_imgs}")
  model    = modelGAN.GAN(img_size, latent_dim)
  model.summary()
  model.train(dataset, num_epochs, num_imgs)
