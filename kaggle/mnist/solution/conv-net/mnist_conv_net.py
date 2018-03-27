import shutil

import os
from datetime import datetime
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.adam import AdamOptimizer

root_dir = '/home/eantaev/prj/ml-just/kaggle/mnist/solution/conv-net/'
root_log_dir = os.path.join(root_dir, 'tf_log')
problem_predictions_path = os.path.join(root_dir, 'test_predictions.csv')

data_dir = '/home/eantaev/prj/ml-just/kaggle/mnist/data/'
train_path = os.path.join(data_dir, 'train.csv')
problem_path = os.path.join(data_dir, 'test.csv')

Dataset = collections.namedtuple('Dataset', ['data', 'target'])


def read_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if 'label' in df.columns:
        return Dataset(data=df.drop('label', axis=1).values.astype(np.float32) / 255,
                       target=df['label'].values)
    else:
        return Dataset(data=df.values.astype(np.float32) / 255,
                       target=None)


k_problem = read_dataset(problem_path)

mnist = input_data.read_data_sets("/tmp/data/mnist")
train = mnist.train
X_train = train.images
y_train = train.labels

n_samples, n_features = X_train.shape
height = 28
width = 28
channels = 1

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps
pool3_stride = 2

n_fc1 = 128
fc1_dropout_rate = 0.5

n_labels = 10

init_learning_rate = 0.001


def generate_run_dir(prefix='', timestamp_suffix=False):
    name = 'run'
    if prefix:
        name = prefix + '-' + name
    if timestamp_suffix:
        name = name + '-' + datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')
    return os.path.join(root_log_dir, name)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var.op.name + '/summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class Run:
    def __init__(self):
        self.graph = None
        self.x = None
        self.y = None
        self.train_step = None
        self.training = None
        self.global_step = None
        self.loss = None
        self.accuracy = None
        self.predict = None
        self.make_predictions = False
        self.saver = None
        self.summary = None
        self.train_summary_writer = None
        self.valid_summary_writer = None
        self.test_summary_writer = None
        self.regularizer = None
        self.hidden_activation = tf.nn.elu
        self.optimizer = AdamOptimizer(learning_rate=init_learning_rate)

        self.run_dir = None
        self.checkpoint_path = None
        self.checkpoint_epoch_path = None
        self.log_train_dir = None
        self.log_valid_dir = None
        self.log_test_dir = None

        self.n_epochs = 40
        self.batch_size = 50
        self.n_batches = n_samples // self.batch_size
        self.continue_training = True

    def init_run_dir(self, run_dir=None):
        print('Run initialization', run_dir)
        self.run_dir = run_dir if run_dir else generate_run_dir()
        self.checkpoint_path = os.path.join(self.run_dir, '10_mnist_dnn_model.ckpt')
        self.checkpoint_epoch_path = self.checkpoint_path + '.epoch'
        self.log_train_dir = os.path.join(self.run_dir, 'train')
        self.log_valid_dir = os.path.join(self.run_dir, 'valid')
        self.log_test_dir = os.path.join(self.run_dir, 'test')
        return self

    def clear(self):
        self.x = None
        self.y = None
        self.train_step = None
        self.global_step = None
        self.loss = None
        self.accuracy = None
        self.accuracy = None
        self.saver = None
        self.summary = None
        self.train_summary_writer = None
        self.valid_summary_writer = None


_run = Run()


def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))


def max_norm_regilarizer(threshold, axes=1, name='max_norm', collection='max_norm'):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, value=clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None

    return max_norm


def construct_kernel_regularizer(run=_run):
    if run.regularizer == 'l2':
        print('using l2 regularizer')
        return tf.contrib.layers.l2_regularizer(0.001)
    if run.regularizer == 'max_norm':
        print('using max_norm regularizer')
        return max_norm_regilarizer(threshold=1.0)
    return None


def construct_graph(run=_run):
    tf.reset_default_graph()

    run.x = tf.placeholder(np.float32, shape=(None, n_features), name='X')
    run.y = tf.placeholder(np.int32, shape=None, name='y')
    run.training = tf.placeholder_with_default(False, shape=(), name='trainig')
    images = tf.reshape(run.x, shape=(-1, height, width, channels))

    he_init = tf.contrib.layers.variance_scaling_initializer()
    kernel_regularizer = construct_kernel_regularizer(run)

    conv1 = tf.layers.conv2d(images, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride,
                             padding=conv1_pad, activation=tf.nn.relu, name='conv1')
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride,
                             padding=conv2_pad, activation=tf.nn.relu, name='conv2')

    with tf.name_scope('max_pooling'):
        fmap_size = height // (conv1_stride * conv2_stride * pool3_stride)
        flat_length = conv2_fmaps * fmap_size * fmap_size
        max_pool = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=pool3_stride)
        max_pool_flat = tf.reshape(max_pool, shape=(-1, flat_length))
        max_pool_out = tf.layers.dropout(max_pool_flat, rate=conv2_dropout_rate,
                                         training=run.training, name='max_pool_out')

    fc1 = tf.layers.dense(max_pool_out, units=n_fc1,
                          activation=run.hidden_activation,
                          kernel_initializer=he_init,
                          kernel_regularizer=kernel_regularizer,
                          name='fc1')
    fc1_drop = tf.layers.dropout(fc1, rate=fc1_dropout_rate)

    logits = tf.layers.dense(fc1_drop, n_labels,
                             kernel_initializer=he_init,
                             # kernel_regularizer=kernel_regularizer,
                             name='logits')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=run.y, logits=logits, name='xentropy')
        mean_xentropy = tf.reduce_mean(xentropy, name='mean_xentropy')
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        run.loss = tf.add_n([mean_xentropy] + regularization_losses, name='loss')
        tf.summary.scalar('loss', run.loss)

    with tf.name_scope('train'):
        run.global_step = tf.Variable(initial_value=0, trainable=False)
        # clip_all_weights = tf.get_collection('max_norm')
        # with tf.control_dependencies(clip_all_weights):
        run.train_step = run.optimizer.minimize(run.loss, global_step=run.global_step, name='train_step')

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, run.y, 1, name='correct')
        run.accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', run.accuracy)

    with tf.name_scope('prediction'):
        run.predict = tf.argmax(logits, axis=1, name='predict')

    run.saver = tf.train.Saver()

    run.graph = tf.get_default_graph()
    for var in run.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        variable_summaries(var)

    run.summary = tf.summary.merge_all()
    run.train_summary_writer = tf.summary.FileWriter(_run.log_train_dir, graph=tf.get_default_graph())
    run.valid_summary_writer = tf.summary.FileWriter(_run.log_valid_dir, graph=tf.get_default_graph())
    run.test_summary_writer = tf.summary.FileWriter(_run.log_test_dir, graph=tf.get_default_graph())

    return run.graph


def init_run_dir(prefix='', timestamp_suffix=False, run=_run):
    run_dir = generate_run_dir(prefix=prefix, timestamp_suffix=timestamp_suffix)
    return run.init_run_dir(run_dir)


def remove_saved_model():
    shutil.rmtree(root_dir)


def remove_current_run():
    shutil.rmtree(_run.run_dir)


def train_session(session, run=_run):
    os.makedirs(run.run_dir, exist_ok=True)
    start_epoch = 0
    if run.continue_training and os.path.isfile(run.checkpoint_epoch_path):
        print('restoring checkpoint')
        run.saver.restore(session, run.checkpoint_path)
        with open(run.checkpoint_epoch_path, 'r') as epoch_file:
            start_epoch = int(epoch_file.readline())
    else:
        session.run(tf.global_variables_initializer())

    print('start epoch', start_epoch)
    # clip_all_weights = tf.get_collection('max_norm')
    for epoch in range(start_epoch, run.n_epochs):
        for step in range(run.n_batches):
            x_batch, y_batch = train.next_batch(run.batch_size)
            session.run(run.train_step, feed_dict={run.x: x_batch,
                                                   run.y: y_batch,
                                                   run.training: True})
            # session.run(clip_all_weights)

        if epoch % 1 == 0:
            run.saver.save(session, save_path=run.checkpoint_path)

            with open(run.checkpoint_epoch_path, 'w') as epoch_file:
                epoch_file.write(str(epoch + 1))
            x_batch, y_batch = train.next_batch(run.batch_size)
            loss_train, acc_train, sum_train, s = session.run(
                [run.loss, run.accuracy, run.summary, run.global_step],
                feed_dict={run.x: x_batch,
                           run.y: y_batch})
            run.train_summary_writer.add_summary(sum_train, global_step=s)
            run.train_summary_writer.flush()

            print('Epoch', epoch, 'step', s,
                  'train {loss:', loss_train, '\tacc:', acc_train, '}')

        if epoch % 5 == 0:
            loss_valid, acc_valid, sum_valid = session.run(
                [run.loss, run.accuracy, run.summary],
                feed_dict={run.x: mnist.validation.images,
                           run.y: mnist.validation.labels})

            run.valid_summary_writer.add_summary(sum_valid, global_step=s)
            run.valid_summary_writer.flush()

            print('Epoch', epoch, 'step', s,
                  'valid {loss:', loss_valid, '\tacc:', acc_valid, '}')

    run.train_summary_writer.close()
    run.valid_summary_writer.close()

    if run.make_predictions:
        # k_problem = read_dataset(problem_path)

        prediction_chunks = [session.run(run.predict, feed_dict={run.x: problem_chunk})
                             for problem_chunk in np.array_split(k_problem.data, 5)]
        predictions = np.concatenate(prediction_chunks)
        # store predictions to file
        pp = pd.DataFrame(data=predictions,
                          index=range(1, predictions.size + 1),
                          columns=['Label'])
        pp.index.name = 'ImageId'

        pp.to_csv(problem_predictions_path)


def train_new_session():
    with tf.Session() as sess:
        train_session(sess, _run)


init_run_dir('r02-batch100-fc128-stride1')

_run.n_epochs = 41
_run.batch_size = 100
_run.n_batches = 100  # n_samples // _run.batch_size
_run.continue_training = True
# _run.regularizer = 'max_norm'
_run.regularizer = None
_run.hidden_activation = tf.nn.relu
# _run.hidden_activation = selu
# _run.optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
_run.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# _run.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
_run.make_predictions = True

construct_graph()
train_new_session()
