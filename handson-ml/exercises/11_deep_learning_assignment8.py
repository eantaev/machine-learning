import tensorflow as tf
import numpy as np
import os
import shutil

root_dir = '/tmp/11_deep_learning_assignment8'
checkpoint_mnist_model_0_4 = os.path.join(root_dir, 'mnist_model_0_4.ckpt')
log_dir = os.path.join(root_dir, 'tf_logs')
train_log_dir = os.path.join(log_dir, 'train')
valid_log_dir = os.path.join(log_dir, 'valid')
best_valid_log_dir = os.path.join(log_dir, 'best_valid')

mnist = tf.contrib.learn.datasets.mnist.load_mnist(root_dir)

n_features = mnist.train.images.shape[1]
n_labels_0_4 = 5
layers = [100, 100, 100, 100, 100]
activation = tf.nn.elu

shutil.rmtree(log_dir, ignore_errors=True)

tf.reset_default_graph()

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=(None, n_features))
    y = tf.placeholder(dtype=tf.int32, shape=(None))

with tf.name_scope('dnn'):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    output = x
    for index, layer_units in enumerate(layers):
        output = tf.layers.dense(inputs=output,
                                 units=layer_units,
                                 activation=activation,
                                 kernel_initializer=he_init,
                                 name='hidden' + str(index))

    logits = tf.layers.dense(inputs=output,
                             units=n_labels_0_4,
                             activation=activation,
                             kernel_initializer=he_init,
                             name='logits')

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1, name='correct')
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(train_log_dir, graph=tf.get_default_graph())
valid_writer = tf.summary.FileWriter(valid_log_dir)
best_valid_writer = tf.summary.FileWriter(best_valid_log_dir)

saver = tf.train.Saver()

n_epochs = 1000
batch_size = 20

x_train_0_4 = mnist.train.images[mnist.train.labels < 5]
y_train_0_4 = mnist.train.labels[mnist.train.labels < 5]

x_validation_0_4 = mnist.validation.images[mnist.validation.labels < 5]
y_validation_0_4 = mnist.validation.labels[mnist.validation.labels < 5]

x_test_0_4 = mnist.test.images[mnist.test.labels < 5]
y_test_0_4 = mnist.test.labels[mnist.test.labels < 5]

best_loss = np.infty
steps_after_best = 0
max_epochs_without_progress = 20

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        n_samples = len(x_train_0_4)
        permutation = np.random.permutation(n_samples)
        for batch_indices in np.array_split(permutation, n_samples // batch_size):
            x_batch = x_train_0_4[batch_indices]
            y_batch = y_train_0_4[batch_indices]
            sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

        train_acc, train_loss, train_summary, step = sess.run(
            [accuracy, loss, summary, global_step],
            feed_dict={x: x_train_0_4, y: y_train_0_4})

        val_acc, val_loss, val_summary = sess.run(
            [accuracy, loss, summary],
            feed_dict={x: x_validation_0_4, y: y_validation_0_4})

        train_writer.add_summary(train_summary, global_step=step)
        train_writer.flush()
        valid_writer.add_summary(val_summary, global_step=step)
        valid_writer.flush()

        print('Epoch', epoch,
              'train_acc', train_acc, 'train_loss', train_loss,
              'val_acc', val_acc, 'val_loss', val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            saver.save(sess, checkpoint_mnist_model_0_4)
            steps_after_best = 0
            best_valid_writer.add_summary(val_summary, global_step=step)
            best_valid_writer.flush()
            print('This is the best val loss so far.')
        else:
            steps_after_best += 1
            print(steps_after_best, 'steps after best.')
            if steps_after_best == 20:
                print('early stopping')
                break

    train_writer.close()
    valid_writer.close()
    best_valid_writer.close()

with tf.Session() as sess:
    saver.restore(sess, checkpoint_mnist_model_0_4)
    val_acc, val_loss = sess.run(
        [accuracy, loss],
        feed_dict={x: x_validation_0_4, y: y_validation_0_4})
    print('Best model:\nval_acc', val_acc, 'val_loss', val_loss)
    test_acc, test_loss = sess.run(
        [accuracy, loss],
        feed_dict={x: x_test_0_4, y: y_test_0_4})
    print('test_acc', test_acc, 'test_loss', test_loss)
