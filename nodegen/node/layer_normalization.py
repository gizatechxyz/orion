import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5): 
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]
    x_mat = np.reshape(X, (row_number, col_number))
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    y_mat = x_diff * inv_std_dev
    Y = np.reshape(y_mat, X_shape) * W + B
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_mean, X_inv_std_dev


def calculate_normalized_shape(X_shape, axis):  
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]


class Layer_normalization(RunAll):
    @staticmethod
    def export4d() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)

            if axis < 0:
                name = f"layer_normalization_4d_axis_negative_{-axis}"
                func_sig = f"input_0.layer_normalization(@input_1,Option::Some(@input_2),Option::Some(IntegerTrait::<i32>::new({-axis}, true)),Option::None,Option::None)"
            else:
                name = f"layer_normalization_4d_axis{axis}"
                func_sig = f"input_0.layer_normalization(@input_1,Option::Some(@input_2),Option::Some(IntegerTrait::<i32>::new({axis}, false)),Option::None,Option::None)"


            x = Tensor(Dtype.FP8x23, X.shape, to_fp(X.flatten(), FixedImpl.FP8x23))
            w = Tensor(Dtype.FP8x23, W.shape, to_fp(W.flatten(), FixedImpl.FP8x23))
            b = Tensor(Dtype.FP8x23, B.shape, to_fp(B.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, Y.shape, to_fp(Y.flatten(), FixedImpl.FP8x23))
            
            make_test([x,w,b], y, func_sig, name)


        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))

    @staticmethod
    def export_default_axis() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        normalized_shape = calculate_normalized_shape(X.shape, -1)
        W = np.random.randn(*normalized_shape).astype(np.float32)
        B = np.random.randn(*normalized_shape).astype(np.float32)
        Y, mean, inv_std_dev = _layer_normalization(X, W, B)

        x = Tensor(Dtype.FP16x16, X.shape, to_fp(X.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, W.shape, to_fp(W.flatten(), FixedImpl.FP16x16))
        b = Tensor(Dtype.FP16x16, B.shape, to_fp(B.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16))
        
        name = "layer_normalization_default_axis"
        make_test([x,w,b], y, "input_0.layer_normalization(@input_1,Option::Some(@input_2),Option::None,Option::None,Option::None)", name)

    @staticmethod
    def export3d_epsilon() -> None:
        epsilon = 1e-1
        X = np.random.randn(2, 3, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis, epsilon)

            if axis < 0:
                name = f"layer_normalization_3d_axis_negative_{-axis}_epsilon"
                func_sig = f"input_0.layer_normalization(@input_1,Option::Some(@input_2),Option::Some(IntegerTrait::<i32>::new({-axis}, true)),Option::Some(FixedTrait::new(6554, false)),Option::None)"
            else:
                name = f"layer_normalization_3d_axis{axis}_epsilon"
                func_sig = f"input_0.layer_normalization(@input_1,Option::Some(@input_2),Option::Some(IntegerTrait::<i32>::new({axis}, false)),Option::Some(FixedTrait::new(6554, false)),Option::None)"

            x = Tensor(Dtype.FP16x16, X.shape, to_fp(X.flatten(), FixedImpl.FP16x16))
            w = Tensor(Dtype.FP16x16, W.shape, to_fp(W.flatten(), FixedImpl.FP16x16))
            b = Tensor(Dtype.FP16x16, B.shape, to_fp(B.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16))
            
            make_test([x,w,b], y, func_sig, name)


        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))

    @staticmethod
    def test_2d_example() -> None:
        X = np.random.randn(3, 4).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis=axis)

            node = onnx.helper.make_node(
                "LayerNormalization",
                inputs=["X", "W", "B"],
                outputs=["Y", "Mean", "InvStdDev"],
                axis=axis,
            )

            x = Tensor(Dtype.FP16x16, X.shape, to_fp(X.flatten(), FixedImpl.FP16x16))
            w = Tensor(Dtype.FP16x16, W.shape, to_fp(W.flatten(), FixedImpl.FP16x16))
            b = Tensor(Dtype.FP16x16, B.shape, to_fp(B.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16))

            name = "layer_normalization_test"
            make_test([x,w,b], y, "input_0.layer_normalization(@input_1,Option::Some(@input_2),Option::None,Option::None,Option::None)", name)

        case(-1)