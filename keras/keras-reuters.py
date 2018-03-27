import numpy as np

from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

NUM_WORDS = 10000

NUM_CLASSES = 46

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=NUM_WORDS)

word_index = reuters.get_word_index()


def vectorize_sequences(sequences, dimension=NUM_WORDS):
    table = np.zeros(shape=(len(sequences), dimension), dtype=np.float32)
    for i, seq in enumerate(sequences):
        table[i, seq] = 1.0
    return table


def train_val_split(x, y):
    return (x[:1000], y[:1000]), (x[1000:], y[1000:])


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# using one-hot encodinf for labels
y_train = to_categorical(train_labels, num_classes=NUM_CLASSES)
y_test = to_categorical(test_labels, num_classes=NUM_CLASSES)

(partial_x_train, partial_y_train), (x_val, y_val) = train_val_split(x_train, y_train)

model = models.Sequential()
model.add(layers.Dense(units=64,
                       activation=activations.relu,
                       kernel_regularizer=regularizers.l2(0.01),
                       input_shape=(NUM_WORDS,)))
model.add(layers.Dense(units=64,
                       activation=activations.relu,
                       kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(units=NUM_CLASSES,
                       activation=activations.softmax))

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(x=partial_x_train, y=partial_y_train,
                    batch_size=512, epochs=20,
                    validation_data=(x_val, y_val))


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


model.predict(x_test)

# labels as int arrays (using sparse cross entropy loss)
y_train = np.asarray(train_labels)
y_test = np.asarray(test_labels)

(partial_x_train, partial_y_train), (x_val, y_val) = train_val_split(x_train, y_train)

model = models.Sequential()
model.add(layers.Dense(units=64, activation=activations.relu, input_shape=(NUM_WORDS,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=64, activation=activations.relu))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=NUM_CLASSES, activation=activations.softmax))
model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(x=partial_x_train, y=partial_y_train,
                    batch_size=512, epochs=20,
                    validation_data=(x_val, y_val))

plot_history(history)


def train_and_plot(m):
    h = m.fit(x=partial_x_train, y=partial_y_train,
              batch_size=512, epochs=20,
              validation_data=(x_val, y_val))
    plot_history(h)
