import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, Trait, FixedImpl
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

    @staticmethod
    def relu_fp8x23():
        x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
        layer = tf.keras.layers.ReLU()
        y = layer(x).numpy()

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "relu_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::relu(@input_0)",
                  name, Trait.NN)

    @staticmethod
    def relu_fp16x16():
        x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
        layer = tf.keras.layers.ReLU()
        y = layer(x).numpy()

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "relu_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::relu(@input_0)",
                  name, Trait.NN)