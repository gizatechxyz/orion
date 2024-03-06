import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl
from .conv import conv


def qlinear_conv(
    x,
    x_scale,
    x_zero_point,
    w,
    w_scale,
    w_zero_point,
    y_scale,
    y_zero_point,
    B=None,
    auto_pad=None,
    dilations=None,
    group=None,
    kernel_shape=None,
    pads=None,
    strides=None,
):
    X = x.astype(np.int32)
    if x_zero_point is not None:
        X -= x_zero_point
    W = w.astype(np.int32)
    if w_zero_point is not None:
        if len(w_zero_point.shape) == 1 and w_zero_point.shape[0] == W.shape[0]:
            missing = (w_zero_point.shape[0],) + (1,) * (len(W.shape) - 1)
            W -= w_zero_point.reshape(missing)
        else:
            W -= w_zero_point
    res = conv(
        X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides
    ).astype(np.int32)
    R = res * (x_scale * w_scale / y_scale)
    if y_zero_point is not None:
        R += y_zero_point
        if y_zero_point.dtype == np.int8:
            R = np.clip(R, -128, 127)
        else:
            R = np.clip(R, 0, 255)
        return (np.rint(R).astype(y_zero_point.dtype),)
    if x.dtype == np.int8:
        R = np.clip(R, -128, 127)
    else:
        R = np.clip(R, 0, 255)
    return (np.rint(R).astype(x.dtype),)


class Qlinear_conv(RunAll):
    @staticmethod
    def export_qlinear_conv() -> None:
        x = np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],

            dtype=np.int8,
        ).reshape((1, 1, 3, 3))
        x_scale = np.float32(0.5)
        x_zero_point = np.int8(2)

        w = np.array([0], dtype=np.int8).reshape((1, 1, 1, 1))
        w_scale = np.array([0.4], dtype=np.float32)
        w_zero_point = np.array([3], dtype=np.int8)

        y_scale = np.float32(0.2)
        y_zero_point = np.int8(4)
        
        param = np.array([0.5, 2, 0.4, 3, 0.2, 4])

        y = qlinear_conv(x,x_scale,x_zero_point,w,w_scale,w_zero_point,y_scale,y_zero_point,)
        y = np.array(y)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        w = Tensor(Dtype.I8, w.shape, w.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())
        param = Tensor(Dtype.FP16x16, param.shape, to_fp(param.flatten(), FixedImpl.FP16x16))
        
        
        name = "qlinear_conv"
        func_sig = "qlinear_conv("
        func_sig += "@input_0,"
        func_sig += "@TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(0)].span(),),"
        func_sig += "@TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(1)].span(),),"
        func_sig += "@input_1,"
        func_sig += "@TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(2)].span(),),"
        func_sig += "@TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(3)].span(),),"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "@TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(4)].span(),),"
        func_sig += "@TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(5)].span(),))"
        make_test(
            [x, w, param], y, func_sig, name) 
