import numpy as np

from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
from keras.datasets import imdb

import matplotlib.pyplot as plt

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

word_index = imdb.get_word_index()
reverse_word_index = {index: word for (word, index) in word_index.items()}


def decode_review(sample):
    return ' '.join((reverse_word_index.get(index - 3, '?') for index in sample))


def vectorize_sequences(sequences, dimension=NUM_WORDS):
    table = np.zeros(shape=(len(sequences), dimension), dtype=np.float32)
    for i, seq in enumerate(sequences):
        table[i, seq] = 1.0
    return table


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype(np.float32)
y_test = np.asarray(test_labels).astype(np.float32)

model = models.Sequential()
model.add(layers.Dense(units=16, activation=activations.relu, input_shape=(NUM_WORDS,)))
model.add(layers.Dense(units=16, activation=activations.relu))
model.add(layers.Dense(units=1, activation=activations.sigmoid))

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(x=partial_x_train,
                    y=partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history

print(history_dict.keys())


def plot_history(history):
    hist_dict = history.history
    loss_values = hist_dict['loss']
    val_loss_values = hist_dict['val_loss']
    acc_values = hist_dict['binary_accuracy']
    val_acc_values = hist_dict['val_binary_accuracy']

    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_values, 'bo')
    plt.plot(epochs, val_loss_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc_values, 'bo')
    plt.plot(epochs, val_acc_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()

plot_history(history)

