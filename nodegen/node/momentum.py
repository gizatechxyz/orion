import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
from typing import List

import numpy as np

#from onnx.reference.ops.op_resize import _get_all_coords

def _run1( r, t, x, g, v, mode="standard", norm_coefficient=None, alpha=None, beta=None):  # type: ignore
    if mode == "standard":
        x_new, v_new = _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)
    else:
        x_new, v_new = _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta)
    return x_new, v_new

def _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply SG with momentum update rule.
    x_new = x - r * v_new
    return x_new, v_new


def _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply Nesterov with momentum update rule.
    x_new = x - r * (g_regularized + alpha * v_new)
    return x_new, v_new

def momentum(*data, alpha=None, beta=None, mode=None, norm_coefficient=None):  # type: ignore
    if len(data) == 5:
        r, t, x, g, v = data
        return _run1(  # type: ignore
            r,
            t,
            x,
            g,
            v,
            norm_coefficient=norm_coefficient,
            alpha=alpha,
            beta=beta,
            mode=mode,
        )
    n = (len(data) - 2) // 3
    xs = []
    vs = []
    for i in range(0, n):
        a, b = _run1(  # type: ignore
            *data[:2], # r and t
            data[2 + i],
            data[2 + n + i],
            data[2 + n * 2 + i],
            norm_coefficient=norm_coefficient,
            alpha=alpha,
            beta=beta,
            mode=mode,
        )
        xs.append(a)
        vs.append(b)
    return tuple(xs + vs)


    
class Momentum(RunAll):
    @staticmethod
    def export_momentum() -> None:
        # Define operator attributes.
        norm_coefficient = 0.001
        alpha = 0.95
        beta = 0.1

        # Define operator inputs.
        r = np.array(0.1, dtype=np.float32)  # scalar
        t = np.array(0, dtype=np.int64)  # scalar
        x = np.array([1.2, 2.8], dtype=np.float32)
        g = np.array([-0.94, -2.5], dtype=np.float32)
        v = np.array([1.7, 3.6], dtype=np.float32)

        # Compute expected outputs of Momentum.
        x_new, v_new = _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)
        
        x = np.array([1.2, 2.8, -0.94, -2.5, 1.7, 3.6])
        param = np.array([r, t, alpha, beta, norm_coefficient])

        x_new = np.array(x_new)
        v_new = np.array(v_new)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        param = Tensor(Dtype.FP16x16, param.shape, to_fp(param.flatten(), FixedImpl.FP16x16))
        x_new = Tensor(Dtype.FP16x16, x_new.shape, to_fp(x_new.flatten(), FixedImpl.FP16x16))
        v_new = Tensor(Dtype.FP16x16, v_new.shape, to_fp(v_new.flatten(), FixedImpl.FP16x16))

        name = "momentum_standard"
        func_sig = "TensorTrait::momentum("
        func_sig += "*input_1.data.at(0),"
        func_sig += "*input_1.data.at(1),"
        func_sig += "@input_0,"
        func_sig += "*input_1.data.at(2),"
        func_sig += "*input_1.data.at(3),"
        func_sig += "MODE::STANDARD,"
        func_sig += "*input_1.data.at(4))"
        make_test(
            [x, param], [x_new, v_new], func_sig, name)
        
    @staticmethod
    def export_nesterov_momentum() -> None:
        # Define operator attributes.
        norm_coefficient = 0.01
        alpha = 0.95
        beta = 1.0

        # Define operator inputs.
        r = np.array(0.1, dtype=np.float32)  # scalar
        t = np.array(0, dtype=np.int64)  # scalar
        x = np.array([1.2, 2.8], dtype=np.float32)
        g = np.array([-0.94, -2.5], dtype=np.float32)
        v = np.array([1.7, 3.6], dtype=np.float32)

        # Compute expected outputs of Momentum.
        x_new, v_new = _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta)

        
        x = np.array([1.2, 2.8, -0.94, -2.5, 1.7, 3.6])
        param = np.array([r, t, alpha, beta, norm_coefficient])

        x_new = np.array(x_new)
        v_new = np.array(v_new)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        param = Tensor(Dtype.FP16x16, param.shape, to_fp(param.flatten(), FixedImpl.FP16x16))
        x_new = Tensor(Dtype.FP16x16, x_new.shape, to_fp(x_new.flatten(), FixedImpl.FP16x16))
        v_new = Tensor(Dtype.FP16x16, v_new.shape, to_fp(v_new.flatten(), FixedImpl.FP16x16))

        name = "momentum_nesterov"
        func_sig = "TensorTrait::momentum("
        func_sig += "*input_1.data.at(0),"
        func_sig += "*input_1.data.at(1),"
        func_sig += "@input_0,"
        func_sig += "*input_1.data.at(2),"
        func_sig += "*input_1.data.at(3),"
        func_sig += "MODE::STANDARD,"
        func_sig += "*input_1.data.at(4))"
        make_test(
            [x, param], [x_new, v_new], func_sig, name)



        
        
