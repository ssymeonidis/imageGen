import os
import time
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import dataset
import modelGeneratorDense as modelGenerator
import modelDiscriminatorDense as modelDiscriminator
import modelGAN
import plotSave

IMG_H = 64
IMG_W = 64
IMG_C = 3

if __name__ == '__main__':
  latent_dim = 64
  num_epochs = 1000
  n_samples  = 100

  g_model = modelGenerator.gen((IMG_H, IMG_W, IMG_C), latent_dim)
  d_model = modelDiscriminator.gen((IMG_H, IMG_W, IMG_C))

  g_model.summary()
  d_model.summary()

  d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9)
  g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9)

  images_dataset = dataset.gen()
  seed = np.random.normal(size=(n_samples, 1, latent_dim))

  for epoch in range(num_epochs):
    start = time.time()
    d_loss = 0.0
    g_loss = 0.0
    for image_batch in images_dataset:
      d_batch_loss, g_batch_loss = modelGAN.train_step(image_batch, latent_dim, g_model, d_model, g_optimizer, d_optimizer)
      d_loss += d_batch_loss
      g_loss += g_batch_loss

    d_loss = d_loss/len(images_dataset)
    g_loss = g_loss/len(images_dataset)

    g_model.save("../models/g_model.keras")
    d_model.save("../models/d_model.keras")

    examples = g_model.predict(seed, verbose=0)
    plotSave.saveNxN(examples, epoch, np.sqrt(n_samples))

    time_taken = time.time() - start
    print(f"[{epoch+1:1.0f}/{num_epochs}] {time_taken:2.2f}s - d_loss: {d_loss:1.4f} - g_loss: {g_loss:1.4f}")

