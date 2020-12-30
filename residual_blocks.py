"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import os  # for os related ops

import tensorflow as tf  # for deep learning based ops
from typing import Tuple


class BasicBlock(tf.keras.layers.Layer):
    """
    Basic Block for residual networks
    """
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int] = (3, 3),
                 padding: str = "same",
                 strides: Tuple[int] = (2, 2),
                 *args,
                 **kwargs):
        """
        Constructor of the model for creating a basic block for the residual network
        """
        super(BasicBlock, self).__init__(*args, **kwargs)

        # first conv block
        self.conv1 = tf.keras.layers.Convolution2D(filters=filters,
                                                   kernel_size=kernel_size,
                                                   strides=strides,
                                                   padding=padding)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU()

        # seconf conv block
        self.conv2 = tf.keras.layers.Convolution2D(filters=filters,
                                                   kernel_size=kernel_size,
                                                   padding=padding,
                                                   strides=(1, 1))
        self.bn2 = tf.keras.layers.BatchNormalization()

        # res conv block
        self.res_conv = tf.keras.layers.Convolution2D(filters=filters,
                                                      kernel_size=(1, 1),
                                                      strides=strides,
                                                      padding=padding)
        self.res_bn = tf.keras.layers.BatchNormalization()

        # final activation
        self.act2 = tf.keras.layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        training = kwargs.get('training', True)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        residual = self.res_conv(inputs)
        residual = self.res_bn(residual)

        output = self.act2(tf.add(x, residual))

        return output
