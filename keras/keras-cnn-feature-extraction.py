import os, shutil
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 150
CAT_FILENAME_PATTERN = 'cat.{}.jpg'
DOG_FILENAME_PATTERN = 'dog.{}.jpg'

base_dir = os.path.expanduser('~/data/kaggle/dogs-vs-cats')
orig_data_dir = os.path.join(base_dir, 'train')

small_dir = os.path.join(base_dir, 'small')
os.makedirs(small_dir, exist_ok=True)

models_dir = os.path.join(small_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

train_dir = os.path.join(small_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
valid_dir = os.path.join(small_dir, 'validation')
os.makedirs(valid_dir, exist_ok=True)
test_dir = os.path.join(small_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.makedirs(train_dogs_dir, exist_ok=True)
train_cats_dir = os.path.join(train_dir, 'cats')
os.makedirs(train_cats_dir, exist_ok=True)
valid_dogs_dir = os.path.join(valid_dir, 'dogs')
os.makedirs(valid_dogs_dir, exist_ok=True)
valid_cats_dir = os.path.join(valid_dir, 'cats')
os.makedirs(valid_cats_dir, exist_ok=True)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.makedirs(test_dogs_dir, exist_ok=True)
test_cats_dir = os.path.join(test_dir, 'cats')
os.makedirs(test_cats_dir, exist_ok=True)


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

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

print(conv_base.summary())


# the last layer of conv_base is
# block5_pool (MaxPooling2D)   (None, 4, 4, 512)


def extract_features(directory, sample_count, batch_size=20):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count,))
    generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        directory,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for input_batch, labels_batch in generator:
        features_batch = conv_base.predict(input_batch)
        features[batch_size * i: batch_size * (i + 1)] = features_batch
        labels[batch_size * i: batch_size * (i + 1)] = labels_batch
        i += 1
        print('batch', i, 'out of', np.ceil(float(sample_count) / batch_size))
        if i * batch_size >= sample_count:
            break
    return features, labels


def extract_features2(directory, sample_count, batch_size=20):
    features = []
    labels = []
    generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        directory,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for input_batch, labels_batch in generator:
        features_batch = conv_base.predict(input_batch)
        features += features_batch
        labels += labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return np.concatenate(features, axis=0), \
           np.concatenate(labels, axis=0)

train_features, train_labels = extract_features(train_dir, 2000)
valid_features, valid_labels = extract_features(valid_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# Train new classifier on extracted features

model = models.Sequential()
model.add(Flatten(input_shape=(4, 4, 512)))
model.add(Dense(256, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(1, activation=sigmoid))

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(valid_features, valid_labels))

# Combine new classifier with conv_base from VGG16

combined = models.Sequential()
combined.add(conv_base)
combined.add(model)
combined.compile(optimizer=optimizers.RMSprop(),
                 loss=losses.binary_crossentropy,
                 metrics=['acc'])

# evaluate it

valid_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    valid_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=20,
    class_mode='binary'
)

combined.evaluate_generator(valid_generator, steps=1)