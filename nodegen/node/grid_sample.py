import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
from .resize import _get_all_coords
import numbers
from typing import List

import numpy as np

#from onnx.reference.ops.op_resize import _get_all_coords

def grid_sample(X, grid, mode='linear', padding_mode='zeros', align_corners=0):
    x_dims = X.shape
    grid_dims = grid.shape
    N = x_dims[0]
    C = x_dims[1]
    y_dims = (N, C, *grid_dims[1:-1])
    if np.prod(y_dims) == 0:
        return np.array([], dtype=X.dtype)
    Y = np.empty(y_dims, dtype=X.dtype)
    for n in range(N):
        grid_data = grid[n]
        for c in range(C):
            X_data = X[n, c]
            num_dims = len(x_dims[2:])
            dims = x_dims[2:]
            border = _prepare_border(dims, align_corners=align_corners)
            for ox in _get_all_coords(Y[n, c]):
                nx = grid_data[tuple(ox)]
                nx = nx[::-1]
                x = _gs_denormalize_coordinates(
                    n=nx, dims=dims, align_corners=align_corners
                )
                if mode == "nearest":
                    x = np.rint(x)
                for i, v in enumerate(x):
                    x_min = border[i]
                    x_max = border[i + num_dims]
                    if v < x_min or v > x_max:
                        if padding_mode == "border":
                            x[i] = _clamp(v, 0, dims[i] - 1)
                        elif padding_mode == "reflection":
                            x[i] = _gs_reflect(v, x_min, x_max)
                if mode == "nearest":
                    x = x.astype(np.int32)
                    Y[n][c][tuple(ox)] = _pixel_at_ndarray(
                        ndarray=X_data,
                        x=x,
                        border=border,
                        padding_mode=padding_mode,
                    )
                    
                elif mode == "linear":
                    Y[n][c][tuple(ox)] = _gs_linear_interpolation_nd_with_x(
                        data=X_data, x=x, border=border, padding_mode=padding_mode
                    )
                elif mode == "cubic":
                    Y[n][c][tuple(ox)] = _gs_cubic_interpolation_nd_with_x(
                        data=X_data, x=x, border=border, padding_mode=padding_mode
                    )
                else:
                    raise RuntimeError(
                        "GridSample interpolation only supports nearest, linear, and cubic modes."
                    )
    return (Y.astype(X.dtype),)


def _gs_denormalize(n, length: int, align_corners: bool):  
    if align_corners:
        x = (n + 1) / 2.0 * (length - 1)
    else:
        x = ((n + 1) * length - 1) / 2.0
    return x

def _gs_denormalize_coordinates(n, dims, align_corners: bool):
    x = np.zeros(len(n), dtype=np.float32)
    for i, (v, dim) in enumerate(zip(n, dims)):
        x[i] = _gs_denormalize(n=v, length=dim, align_corners=align_corners)
    return x

def _gs_reflect(x, x_min, x_max):  # type: ignore
    """Reflect by the near border till within the borders
    Use float for borders to avoid potential issues with integer T
    """
    fx = x
    rng = x_max - x_min
    if fx < x_min:           
        dx = x_min - fx
        n = int(dx / rng)
        r = dx - n * rng
        if n % 2 == 0:
            fx = x_min + r
        else:
            fx = x_max - r
    elif fx > x_max:
        dx = fx - x_max
        n = int(dx / rng)
        r = dx - n * rng
        if n % 2 == 0:
            fx = x_max - r
        else:
            fx = x_min + r
    return fx

def _gs_get_cubic_coeffs(x, coeffs):  # type: ignore
    """Calculate cubic convolution interpolation coefficients
    ROBERT G. KEYS https://ieeexplore.ieee.org/document/1163711
    Use float to avoid potential issues with integer.
    """
    cubic_alpha = -0.75
    x = abs(x)
    coeffs[0] = (
        (cubic_alpha * (x + 1) - 5 * cubic_alpha) * (x + 1) + 8 * cubic_alpha
    ) * (x + 1) - 4 * cubic_alpha
    coeffs[1] = ((cubic_alpha + 2) * x - (cubic_alpha + 3)) * x * x + 1
    coeffs[2] = ((cubic_alpha + 2) * (1 - x) - (cubic_alpha + 3)) * (1 - x) * (
        1 - x
    ) + 1
    coeffs[3] = (
        (cubic_alpha * (2 - x) - 5 * cubic_alpha) * (2 - x) + 8 * cubic_alpha
    ) * (2 - x) - 4 * cubic_alpha
    
def _gs_get_linear_coeffs(x, coeffs):
    x = abs(x)
    coeffs[0] = 1 - x
    coeffs[1] = x
    
def _gs_bicubic_interpolate(p, x, y):  # type: ignore
    v = np.empty((4,), dtype=p.dtype)
    coeffs = np.empty((4,), dtype=p.dtype)
    _gs_get_cubic_coeffs(x, coeffs)
    for i in range(4):
        v[i] = coeffs @ p[i, :]
    _gs_get_cubic_coeffs(y, coeffs)
    return coeffs @ v

def _gs_cubic_interpolation_1d_with_x(data, x, border, padding_mode):
    v = np.empty((4,), dtype=data.dtype)
    coeffs = np.empty((4,), dtype=data.dtype)
    x_0 = int(np.floor(x))
    x_1 = x_0 + 1
    x_2 = x_0 + 2
    x_minus_1 = x_0 - 1
    _gs_get_cubic_coeffs(x - x_0, coeffs)
    v[0] = _pixel_at_array(
        array=data, i=x_minus_1, border=border, padding_mode=padding_mode
    )
    v[1] = _pixel_at_array(
        array=data, i=x_0, border=border, padding_mode=padding_mode
    )
    v[2] = _pixel_at_array(
        array=data, i=x_1, border=border, padding_mode=padding_mode
    )
    v[3] = _pixel_at_array(
        array=data, i=x_2, border=border, padding_mode=padding_mode
    )
    return coeffs @ v

def _gs_linear_interpolation_1d_with_x(data, x, border, padding_mode):
    v = np.empty((2,), dtype=data.dtype)
    coeffs = np.empty((2,), dtype=data.dtype)
    x_0 = int(np.floor(x))
    x_1 = x_0 + 1
    _gs_get_linear_coeffs(x - x_0, coeffs)
    v[0] = _pixel_at_array(
        array=data, i=x_0, border=border, padding_mode=padding_mode
    )
    v[1] = _pixel_at_array(
        array=data, i=x_1, border=border, padding_mode=padding_mode
    )
    return coeffs @ v

def _gs_linear_interpolation_nd_with_x(data, x, border, padding_mode):
    num_dims = data.ndim
    assert num_dims == len(x) == int(len(border) / 2)
    if num_dims == 1:
        return _gs_linear_interpolation_1d_with_x(
            data=data, x=x[0], border=border, padding_mode=padding_mode
        )
    res1d = []
    for i in range(data.shape[0]):
        r = _gs_linear_interpolation_nd_with_x(
            data=data[i],
            x=x[1:],
            border=list(border[1:num_dims])
            + list(border[1 + num_dims : 2 * num_dims]),
            padding_mode=padding_mode,
        )
        res1d.append(r)
    res1d = np.array(res1d)
    return _gs_linear_interpolation_1d_with_x(
        data=res1d,
        x=x[0],
        border=[border[0], border[num_dims]],
        padding_mode=padding_mode,
    )
    
def _gs_cubic_interpolation_nd_with_x(data, x, border, padding_mode):
    num_dims = data.ndim
    assert num_dims == len(x) == int(len(border) / 2)
    if num_dims == 1:
        return _gs_cubic_interpolation_1d_with_x(
            data=data, x=x[0], border=border, padding_mode=padding_mode
        )
    res1d = []
    for i in range(data.shape[0]):
        r = _gs_cubic_interpolation_nd_with_x(
            data=data[i],
            x=x[1:],
            border=list(border[1:num_dims])
            + list(border[1 + num_dims : 2 * num_dims]),
            padding_mode=padding_mode,
        )
        res1d.append(r)
    res1d = np.array(res1d)
    return _gs_cubic_interpolation_1d_with_x(
        data=res1d,
        x=x[0],
        border=[border[0], border[num_dims]],
        padding_mode=padding_mode,
    )
    
def _clamp(val, lo, hi):  # type: ignore
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

def _pixel_at_ndarray(ndarray, x: List, border, padding_mode):  # type: ignore
    # boarder: [x_1_min, x_2_min, ..., x_1_max, x_2_max, ...]
    num_dims = ndarray.ndim
    assert num_dims == len(x) == int(len(border) / 2)
    if num_dims == 1:
        return _pixel_at_array(
            array=ndarray, i=x[0], border=border, padding_mode=padding_mode
        )
    i = x[0]
    d = ndarray.shape[0]
    if padding_mode == "zeros":
        if i >= 0 and i < d:
            ndarray = ndarray[i]
        else:
            # Trick
            i = 0
            ndarray = np.zeros_like(ndarray[i])
    elif padding_mode == "border":
        i = _clamp(i, 0, d - 1)
        ndarray = ndarray[i]
    else: 
        i = int(_gs_reflect(i, border[0], border[num_dims]))
        ndarray = ndarray[i]
    return _pixel_at_ndarray(
        ndarray=ndarray,
        x=x[1:],
        border=list(border[1:num_dims]) + list(border[1 + num_dims : 2 * num_dims]),
        padding_mode=padding_mode,
    )
def _pixel_at_array(array, i: int, border, padding_mode):  # type: ignore
    assert array.ndim == 1
    d = array.shape[0]
    if padding_mode == "zeros":
        if i >= 0 and i < d:
            pixel = array[i]
        else:
            pixel = 0
    elif padding_mode == "border":
        i = _clamp(i, 0, d - 1)
        pixel = array[i]
    else: 
        i = int(_gs_reflect(i, border[0], border[1]))
        pixel = array[i]
    return pixel

def _prepare_border(dims, align_corners: bool):
    # boarder: [x_1_min, x_2_min, ..., x_1_max, x_2_max, ...]
    num_dims = len(dims)
    borders = np.zeros(num_dims * 2)
    for i in range(num_dims):
        # min
        borders[i] = -0.5
        # max
        borders[i + num_dims] = dims[i] - 0.5
        if align_corners:
            # min
            borders[i] = 0.0
            # max
            borders[i + num_dims] = dims[i] - 1.0
    return borders


class Grid_sample(RunAll):
    
    @staticmethod
    def export_gridsample() -> None:
        x = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        
        grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.6000, -1.0000],
                        [-0.2000, -1.0000],
                        [0.2000, -1.0000],
                        [0.6000, -1.0000],
                        [1.0000, -1.0000],
                    ],
                    [
                        [-1.0000, -0.6000],
                        [-0.6000, -0.6000],
                        [-0.2000, -0.6000],
                        [0.2000, -0.6000],
                        [0.6000, -0.6000],
                        [1.0000, -0.6000],
                    ],
                    [
                        [-1.0000, -0.2000],
                        [-0.6000, -0.2000],
                        [-0.2000, -0.2000],
                        [0.2000, -0.2000],
                        [0.6000, -0.2000],
                        [1.0000, -0.2000],
                    ],
                    [
                        [-1.0000, 0.2000],
                        [-0.6000, 0.2000],
                        [-0.2000, 0.2000],
                        [0.2000, 0.2000],
                        [0.6000, 0.2000],
                        [1.0000, 0.2000],
                    ],
                    [
                        [-1.0000, 0.6000],
                        [-0.6000, 0.6000],
                        [-0.2000, 0.6000],
                        [0.2000, 0.6000],
                        [0.6000, 0.6000],
                        [1.0000, 0.6000],
                    ],
                    [
                        [-1.0000, 1.0000],
                        [-0.6000, 1.0000],
                        [-0.2000, 1.0000],
                        [0.2000, 1.0000],
                        [0.6000, 1.0000],
                        [1.0000, 1.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )
        
        y = grid_sample(x, grid, mode ="linear")
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_gridsample_paddingmode_zeros() -> None:
        x = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        grid = np.array(
            [
                [
                    [
                        [-10.0000, -10.0000],
                        [-5.0000, -5.0000],
                        [-0.2000, -0.2000],
                        [10.0000, 10.0000],
                    ],
                    [
                        [10.0000, 10.0000],
                        [-0.2000, -0.2000],
                        [5.0000, 5.0000],
                        [10.0000, 10.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        y = grid_sample(x, grid, mode ="linear")
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample_padding_zeros"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_gridsample_paddingmode_border() -> None:
        x = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        grid = np.array(
            [
                [
                    [
                        [-10.0000, -10.0000],
                        [-5.0000, -5.0000],
                        [-0.2000, -0.2000],
                        [10.0000, 10.0000],
                    ],
                    [
                        [10.0000, 10.0000],
                        [-0.2000, -0.2000],
                        [5.0000, 5.0000],
                        [10.0000, 10.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        y = grid_sample(x, grid, mode ="linear", padding_mode="border")
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample_padding_border"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(PADDING_MODE::BORDER))"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_gridsample_paddingmode_reflection() -> None:
        x = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        grid = np.array(
            [
                [
                    [
                        [-10.0000, -10.0000],
                        [-5.0000, -5.0000],
                        [-0.2000, -0.2000],
                        [10.0000, 10.0000],
                    ],
                    [
                        [10.0000, 10.0000],
                        [-0.2000, -0.2000],
                        [5.0000, 5.0000],
                        [10.0000, 10.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )
        
        y = grid_sample(x, grid, mode ="linear", padding_mode="reflection")
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample_padding_reflection"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(PADDING_MODE::REFLECTION))"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_gridsample_mode_aligncorners() -> None:
        x = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        y = grid_sample(x, grid, mode ="linear", align_corners=1)
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample_aligncorners"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::Some(1),"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
              
        
    @staticmethod
    def export_gridsample_nearest() -> None:
        x = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        y = grid_sample(x, grid, mode ="nearest", align_corners=0)
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample_nearest"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::Some(0),"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None)"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
        
  
    @staticmethod
    def export_gridsample_nearest_align_corner() -> None:
        x = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        y = grid_sample(x, grid, mode ="nearest", align_corners=1)
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample_nearest_aligncorner"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::Some(1),"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None)"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_gridsample_cubic() -> None:
        x = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        y = grid_sample(x, grid, mode ="cubic", align_corners=0)
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        grid = Tensor(Dtype.FP16x16, grid.shape, to_fp(grid.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "grid_sample_cubic"
        func_sig = "NNTrait::grid_sample("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::Some(0),"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None)"
        make_test(
            [x, grid], y, func_sig, name, Trait.NN)
