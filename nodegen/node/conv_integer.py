import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, Tensor, Dtype, Trait
from .conv import conv

import numpy as np

def conv_integer( 
    X,
    W,
    x_zero_point=None,
    w_zero_point=None,
    auto_pad=None,
    dilations=None,
    group=None,
    kernel_shape=None,
    pads=None,
    strides=None,
):
    if len(X.shape) < 3:
        raise ValueError(
            f"X must have at least 3 dimensions but its shape is {X.shape}."
        )
    X = X.astype(np.int32)
    if x_zero_point:
        X -= x_zero_point
    W = W.astype(np.int32)
    if w_zero_point:
        W -= w_zero_point
    return (
        conv(
            X, W, None, auto_pad, dilations, group, kernel_shape, pads, strides
        ).astype(np.int32),
    )


class Conv_integer(RunAll):

    @staticmethod
    def export_without_padding() -> None:
        x = (
            np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
            .astype(np.uint8)
            .reshape((1, 1, 3, 3))
        )
        x_zero_point = (
            np.array([1])
            .astype(np.uint8)
            .reshape((1, 1, 1, 1))
        )
        w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))


        y = conv_integer(x,w, x_zero_point)
        y = np.array(y[0])

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        x_zero_point = Tensor(Dtype.I8, x_zero_point.shape, x_zero_point.flatten())
        w = Tensor(Dtype.I8, w.shape, w.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "conv_interger_no_padding"
        func_sig = "NNTrait::conv_integer("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::Some(@input_2),"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w, x_zero_point], y, func_sig, name, Trait.NN)  
    
    @staticmethod
    def export_with_padding() -> None:
        x = (
            np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
            .astype(np.uint8)
            .reshape((1, 1, 3, 3))
        )
        x_zero_point = (
            np.array([1])
            .astype(np.uint8)
            .reshape((1, 1, 1, 1))
        )
        w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))

        y = conv_integer(x,w, x_zero_point=x_zero_point, pads=[1, 1, 1, 1])
        y = np.array(y[0])

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        x_zero_point = Tensor(Dtype.I8, x_zero_point.shape, x_zero_point.flatten())
        w = Tensor(Dtype.I8, w.shape, w.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "conv_interger_with_padding"
        func_sig = "NNTrait::conv_integer("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::Some(@input_2),"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1, 1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x, w, x_zero_point], y, func_sig, name, Trait.NN)  
        


        