from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float64,
                 reshape=False):
        """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float64):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            if len(images.shape) == 4:
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2] * images.shape[3])
        if dtype == dtypes.float64:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float64)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _normalize_meanstd(self, a, axis=None):
        # axis param denotes axes along which mean & std reductions are to be performed
        mean = np.mean(a, axis=axis, keepdims=True)
        std = np.sqrt(((a - mean) ** 2).mean(axis=axis, keepdims=True))
        return (a - mean) / std

    def _per_image_standardization(self, image):
        return self._normalize_meanstd(image, axis=(1, 2))

    def _crop_image(self, image, target_height, target_width):
        _, height, width, _ = image.shape
        width_diff = width - target_width
        offset_crop_width = max(width_diff // 2, 0)

        height_diff = height - target_height
        offset_crop_height = max(height_diff // 2, 0)

        cropped = image[:, offset_crop_height:-offset_crop_height, offset_crop_width:-offset_crop_width, :]
        return cropped

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def crop_image(image, target_height, target_width):
    _, height, width, _ = image.shape
    width_diff = width - target_width
    offset_crop_width = max(width_diff // 2, 0)

    height_diff = height - target_height
    offset_crop_height = max(height_diff // 2, 0)

    cropped = image[:, offset_crop_height:-offset_crop_height, offset_crop_width:-offset_crop_width, :]
    return cropped


def per_image_standardization(image):
    return normalize_meanstd(image, axis=(1, 2))


def normalize_meanstd(a, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean) ** 2).mean(axis=axis, keepdims=True))
    return (a - mean) / std


def load_cifar10(dtype=dtypes.float64, reshape=True, crop_size=32):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    train_images = per_image_standardization(crop_image(train_images, crop_size, crop_size))
    test_images = per_image_standardization(crop_image(test_images, crop_size, crop_size))

    options = dict(dtype=dtype, reshape=reshape)

    train = DataSet(train_images, train_labels, **options)
    test = DataSet(test_images, test_labels, **options)
    validation = DataSet(train_images[:0], train_labels[:0], **options)

    return base.Datasets(train=train, validation=validation, test=test)
