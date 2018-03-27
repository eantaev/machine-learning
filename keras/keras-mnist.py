import numpy as np

from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

NUM_CLASSES = 10


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


def train_and_plot(m):
    h = m.fit(x=partial_x_train, y=partial_y_train,
              batch_size=512, epochs=20,
              validation_data=(x_val, y_val))
    plot_history(h)


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

x_train = train_data.reshape((-1, 28, 28, 1)).astype(np.float32) / 255
x_test = test_data.reshape((-1, 28, 28, 1)).astype(np.float32) / 255

# using one-hot encodinf for labels
# y_train = to_categorical(train_labels, num_classes=NUM_CLASSES)
# y_test = to_categorical(test_labels, num_classes=NUM_CLASSES)

y_train = np.asarray(train_labels).astype(np.float32)
y_test = np.asarray(test_labels).astype(np.float32)


def train_val_split(x, y, val_samples=1000):
    return (x[val_samples:], y[val_samples:]), (x[:val_samples], y[:val_samples])


(partial_x_train, partial_y_train), (x_val, y_val) = train_val_split(x_train, y_train, val_samples=5000)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activations.relu, input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu))
model.add(layers.Flatten())
model.add(layers.Dense(units=64, activation=activations.relu))
model.add(layers.Dense(units=10, activation=activations.softmax))

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(x=partial_x_train, y=partial_y_train,
                    batch_size=64, epochs=20,
                    validation_data=(x_val, y_val))


datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant')

i = 0
for batch in datagen.flow(x_train, y_train, batch_size=1):
    plt.figure(i)
    plt.imshow(batch[0].reshape(28, 28), cmap='gray')
    i += 1
    if i % 4 == 0:
        break
    plt.show()

model.load_weights()