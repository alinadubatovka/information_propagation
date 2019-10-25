import math
import sys
import time

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.initializers import Initializer


class GSM(Initializer):
    """Initializer that generates tensors with a normal distribution.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed`
        for behavior.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_init = tf.initializers.random_normal(mean=mean, stddev=stddev, seed=seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported.
        Raises:
          ValueError: If the dtype is not floating point
          :param **kwargs:
        """
        new_shape = [dim // 2 for dim in shape]
        W_0 = self._random_init(new_shape, dtype)
        return tf.concat([tf.concat([W_0, tf.negative(W_0)], axis=0), tf.concat([tf.negative(W_0), W_0], axis=0)],
                         axis=1)

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed
        }


class Ortho(Initializer):
    """Initializer that generates tensors with a normal distribution.
    Args:
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed`
        for behavior.
    """

    def __init__(self, seed=None):
        self.seed = seed
        self._random_init = tf.initializers.orthogonal(seed=seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported.
        Raises:
          ValueError: If the dtype is not floating point
          :param **kwargs:
        """
        new_shape = [dim // 2 for dim in shape]
        W_0 = self._random_init(new_shape, dtype)
        return tf.concat([tf.concat([W_0, tf.negative(W_0)], axis=0), tf.concat([tf.negative(W_0), W_0], axis=0)],
                         axis=1)

    def get_config(self):
        return {"seed": self.seed}


def train_configuration(N, L, sigma_w, sigma_b, init_mode, dataset, num_epoch, dataset_name):
    kernel_initializer = {
        "GSM": GSM(stddev=sigma_w),
        "He": tf.initializers.random_normal(mean=0.0, stddev=sigma_w),
        "Ortho": Ortho(),
    }[init_mode]
    print(kernel_initializer)
    bias_initializer = tf.initializers.random_normal(stddev=sigma_b)

    num_classes = dataset.train.labels.shape[1]
    input_size = dataset.train.images.shape[1]

    input = tf.placeholder(tf.float64, [None, input_size])
    labels = tf.placeholder(tf.float64, [None, num_classes])

    hidden = tf.layers.Dense(N, activation=tf.nn.relu,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer)(input)
    for _ in range(L):
        hidden = tf.layers.Dense(N, activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer)(hidden)
    # Last layer
    logits = tf.layers.Dense(num_classes, activation=tf.nn.relu,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer)(hidden)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    batch_size = 100
    batch_cnt = dataset.train.num_examples // batch_size
    num_steps = num_epoch * batch_cnt

    step = tf.placeholder(tf.int32)

    if dataset_name == "cifar10":
        # learning rate for CIFAR-10
        lr = (0.00001 + tf.train.exponential_decay(0.0005, step, num_steps, 1 / math.e)) / (L + 1)
    else:
        # learning rate for MNIST
        lr = (0.0001 + tf.train.exponential_decay(0.003, step, 2 * num_epoch * batch_cnt, 1 / math.e)) / (L + 1)

    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    test_acc = []
    start_time = time.time()

    for epoch in range(num_epoch):
        epoch_start = time.time()
        test_acc_epoch = []
        for batch_num in range(batch_cnt):
            batch_features, batch_labels = dataset.train.next_batch(batch_size)
            sess.run(train_step,
                     feed_dict={input: batch_features, labels: batch_labels, step: epoch * batch_cnt + batch_num})

            if batch_num % 100 == 0:
                acc = sess.run(accuracy, feed_dict={input: dataset.test.images, labels: dataset.test.labels})
                test_acc_epoch.append(acc)

        epoch_end = time.time()
        test_acc.append(max(test_acc_epoch))

        print("Epoch # %d: max test accuracy: %f, time: %f" % (epoch, max(test_acc_epoch), epoch_end - epoch_start))
        sys.stdout.flush()
    end_time = time.time()

    return max(test_acc), end_time - start_time
