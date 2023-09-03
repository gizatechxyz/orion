import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
import tensorflow as tf


class Leaky_relu(RunAll):

    @staticmethod
    def leaky_relu_fp8x23():

        x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
        layer = tf.keras.layers.LeakyReLU(alpha=0.1)
        y = layer(x).numpy()

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "leaky_relu_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::leaky_relu(@input_0, @FixedTrait::new(838861, false))",
                  name, Trait.NN)

    @staticmethod
    def leaky_relu_fp16x16():

        x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
        layer = tf.keras.layers.LeakyReLU(alpha=0.1)
        y = layer(x).numpy()

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "leaky_relu_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::leaky_relu(@input_0, @FixedTrait::new(6554, false))",
                  name, Trait.NN)
