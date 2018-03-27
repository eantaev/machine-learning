import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
import collections

root_dir = '/home/eantaev/prj/ml-just/kaggle/mnist/feed-forward-dnn'
train_path = os.path.join(root_dir, 'train.csv')
problem_path = os.path.join(root_dir, 'test.csv')
problem_predictions_path = os.path.join(root_dir, 'test_predictions.csv')

tf_model_path = os.path.join(root_dir, 'tf_log/adam001_elu_max_norm_reg-run')
tf_checkpoint_path = os.path.join(tf_model_path,  '10_mnist_dnn_model.ckpt')
tf_meta_path = tf_checkpoint_path + '.meta'

Dataset = collections.namedtuple('Dataset', ['data', 'target'])


def read_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if 'label' in df.columns:
        return Dataset(data=df.drop('label', axis=1).values.astype(np.float32) / 255,
                       target=df['label'].values)
    else:
        return Dataset(data=df.values.astype(np.float32) / 255,
                       target=None)


k_train = read_dataset(train_path)
k_problem = read_dataset(problem_path)

# tf_mnist = tf.contrib.learn.datasets.mnist.load_mnist(root_dir)

with tf.Session() as sess:
    # restore tf model from tf_model_path
    saver = tf.train.import_meta_graph(tf_meta_path)
    saver.restore(sess, tf_checkpoint_path)
    g = tf.get_default_graph()
    predict = g.get_tensor_by_name('prediction/predict:0')
    x = g.get_tensor_by_name('X:0')
    y = g.get_tensor_by_name('y:0')
    # predict test dataset
    problem_predictions = sess.run(predict, feed_dict={x: k_problem.data})

# store predictions to file
pp = pd.DataFrame(data=problem_predictions,
                  index=range(1, problem_predictions.size + 1),
                  columns=['Label'])
pp.index.name = 'ImageId'

pp.to_csv(problem_predictions_path)