import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, Tensor, Dtype, Trait
import tensorflow as tf


class Relu(RunAll):

    @staticmethod
    def relu_i32():
        x = np.random.randint(-5, 9, (2, 2)).astype(np.int32)
        layer = tf.keras.layers.ReLU()
        y = layer(x).numpy()

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "relu_i32"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::relu(@input_0)",
                  name, Trait.NN)

    @staticmethod
    def relu_i8():
        x = np.random.randint(-5, 9, (2, 2)).astype(np.int8)
        layer = tf.keras.layers.ReLU()
        y = layer(x).numpy()

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "relu_i8"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::relu(@input_0)",
                  name, Trait.NN)
