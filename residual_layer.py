from typing import Tuple  # for the typing
import tensorflow as tf  # for deep learning
from typing import List  # for typings


class BatchNormRelu(tf.keras.layers.Layer):
    def __init__(self, channel_axis: int = -1, *args, **kwargs) -> None:
        super(BatchNormRelu, self).__init__(*args, **kwargs)
        self.channel_axis = channel_axis

    def build(self, input_shape: List[int]) -> None:
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.activation(self.batch_norm(inputs))


class Conv2DBnRelu(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = "same",
                 reverse: bool = False,
                 activate: bool = True,
                 *args,
                 **kwargs) -> None:
        super(Conv2DBnRelu, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.reverse = reverse
        self.activate = activate

    def build(self, input_shape) -> None:
        self.conv2d = tf.keras.layers.Conv2D(filters=self.filters,
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding)
        self.bn_activation = BatchNormRelu()

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        if not self.activate:
            return self.conv2d(inputs)

        # perform bn->relu->conv op
        if self.reverse:
            return self.conv2d(self.bn_activation(inputs))

        # perform conv->bn->relu op
        return self.bn_activation(self.conv2d(inputs))


class ShortCut(tf.keras.layers.Layer):

    # constructor
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = "same",
                 *args,
                 **kwargs) -> None:
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        super(ShortCut, self).__init__(*args, **kwargs)

    # build method
    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.conv = tf.keras.layers.Conv2D(filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding=self.padding)
        self.add = tf.keras.layers.Add()

    # call method
    def call(self, inputs, residual, **kwargs) -> tf.Tensor:
        shortcut = inputs

        # shortcuts
        shortcut = self.conv(inputs)

        # return merged layer
        return self.add([shortcut, residual])


class ResidualLayer2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int = 64,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = "same",
                 *args,
                 **kwargs) -> None:

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        super().__init__(*args, **kwargs)

    def build(self, input_shape) -> None:
        self.activation1 = BatchNormRelu()
        self.conv1 = Conv2DBnRelu(filters=self.filters,
                                  strides=self.strides,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding)
        self.conv2 = Conv2DBnRelu(filters=self.filters,
                                  strides=(1, 1),
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  activate=False)
        self.shortcut = ShortCut(filters=self.filters,
                                 strides=self.strides,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding)
        self.activation2 = BatchNormRelu()

    def call(self, inputs, **kwargs) -> None:
        inputs = self.activation1(inputs)
        x = self.conv1(inputs)
        x = self.conv2(inputs)
        x = self.shortcut(inputs, x)

        return self.activation2(x)