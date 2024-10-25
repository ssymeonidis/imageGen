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
import cv2
import numpy as np

# save sample image
def saveNxN(examples, epoch, n):
  n = int(n)
  examples = (examples + 1) / 2.0
  examples = examples * 255
  filename = f"../samples/generated_plot_epoch-{epoch+1}.png"
  cat_image = None
  for i in range(n):
    start_idx = i*n
    end_idx = (i+1)*n
    image_list = examples[start_idx:end_idx]
    if i== 0:
      cat_image = np.concatenate(image_list, axis=1)
    else:
      tmp = np.concatenate(image_list, axis=1)
      cat_image = np.concatenate([cat_image, tmp], axis=0)
  cv2.imwrite(filename, cat_image)

