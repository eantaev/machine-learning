# start notebook with command:
# jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras import models
from keras import activations
from keras.activations import relu, sigmoid
from keras import losses
from keras import regularizers
from keras import optimizers

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16
from keras.preprocessing import image

IMAGE_SIZE = 150
CAT_FILENAME_PATTERN = 'cat.{}.jpg'
DOG_FILENAME_PATTERN = 'dog.{}.jpg'

base_dir = os.path.expanduser('~/data/kaggle/dogs-vs-cats')
orig_data_dir = os.path.join(base_dir, 'train')

small_dir = os.path.join(base_dir, 'small')

models_dir = os.path.join(small_dir, 'models')

train_dir = os.path.join(small_dir, 'train')
valid_dir = os.path.join(small_dir, 'validation')
test_dir = os.path.join(small_dir, 'test')

train_dogs_dir = os.path.join(train_dir, 'dogs')
train_cats_dir = os.path.join(train_dir, 'cats')
valid_dogs_dir = os.path.join(valid_dir, 'dogs')
valid_cats_dir = os.path.join(valid_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')


def plot_history(history):
    hist_dict = history.history
    loss_values = hist_dict['loss']
    val_loss_values = hist_dict['val_loss']
    acc_values = hist_dict['acc']
    val_acc_values = hist_dict['val_acc']

    epochs = range(1, len(loss_values) + 1)

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,
                                   sharex='all', figsize=(10, 7))
    ax0.plot(epochs, loss_values, 'bo')
    ax0.plot(epochs, val_loss_values, 'b+')
    ax0.set_ylabel('Loss')

    ax1.plot(epochs, acc_values, 'bo')
    ax1.plot(epochs, val_acc_values, 'b+')
    ax1.set_ylabel('Accuracy')

    plt.show()


vgg16 = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

img_path = os.path.join(test_cats_dir, CAT_FILENAME_PATTERN.format(1700))

img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

print(img_tensor.shape)

plt.imshow(img_tensor[0])

# Build activation model

layer_name = 'block3_conv1'
filter_index = 0

import keras.backend as K

layer_output = vgg16.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])
grads = K.gradients(loss, vgg16.input)[0]
# trick for improving learning speed
grads /= K.sqrt(K.mean(K.square(grads)) + 1e-5)

iterate = K.function([vgg16.input], [loss, grads])

loss_value, grads_vaue = iterate([np.random.rand(1, 150, 150, 3)])

input_data_image = np.random.rand(1, IMAGE_SIZE, IMAGE_SIZE, 3) * 20 + 128

step = 1
for i in range(40):
    loss_value, grads_value = iterate([input_data_image])
    input_data_image += grads_value * step
    print('Step', i, 'out of', 40)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to rgb
    x *= 255
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def generate_pattern(layer_name, filter_index, size=IMAGE_SIZE):
    layer_output = vgg16.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, vgg16.input)[0]
    grads /= K.sqrt(K.mean(K.square(grads)) + 1e-5)

    iterate = K.function([vgg16.input], [loss, grads])

    input_data_image = np.random.rand(1, size, size, 3) * 20 + 128
    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_data_image])
        input_data_image += grads_value * step

    img = input_data_image[0]
    return deprocess_image(img)
