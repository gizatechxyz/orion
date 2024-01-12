# Python test implementation from ONNX library : https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_resize.py

import numpy as np
from typing import Any, Callable

from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


def _cartesian(arrays: list[np.ndarray], out: np.ndarray | None = None) -> np.ndarray:
    #From https://stackoverflow.com/a/1235363
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        _cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def _get_neighbor_idxes(x: float, n: int, limit: int) -> np.ndarray:
    idxes = sorted(range(limit), key=lambda idx: (abs(x - idx), idx))[:n]
    idxes = sorted(idxes)
    return np.array(idxes)


def _get_neighbor(x: float, n: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    pad_width = np.ceil(n / 2).astype(int)
    padded = np.pad(data, pad_width, mode="edge")
    x += pad_width

    idxes = _get_neighbor_idxes(x, n, len(padded))


    ret = padded[idxes]
    return idxes - pad_width, ret

def linear_coeffs(ratio: float, scale: float | None = None) -> np.ndarray:
    del scale  
    return np.array([1 - ratio, ratio])


def linear_coeffs_antialias(ratio: float, scale: float) -> np.ndarray:
    scale = min(scale, 1.0)

    start = int(np.floor(-1 / scale) + 1)
    footprint = 2 - 2 * start
    args = (np.arange(start, start + footprint) - ratio) * scale
    coeffs = np.clip(1 - np.abs(args), 0, 1)

    return np.array(coeffs) / sum(coeffs) 

def cubic_coeffs_antialias(ratio: float, scale: float, A: float = -0.75) -> np.ndarray:
    scale = min(scale, 1.0)

    def compute_coeff(x: float) -> float:
        x = abs(x)
        x_2 = x * x
        x_3 = x * x_2
        if x <= 1:
            return (A + 2) * x_3 - (A + 3) * x_2 + 1
        if x < 2:
            return A * x_3 - 5 * A * x_2 + 8 * A * x - 4 * A
        return 0.0

    i_start = int(np.floor(-2 / scale) + 1)
    i_end = 2 - i_start
    args = [scale * (i - ratio) for i in range(i_start, i_end)]
    coeffs = [compute_coeff(x) for x in args]
    return np.array(coeffs) / sum(coeffs)

def nearest_coeffs(
    ratio: float | int | np.ndarray, mode: str = "round_prefer_floor"
) -> np.ndarray:
    if isinstance(ratio, int) or ratio.is_integer():
        return np.array([0, 1])
    if mode == "round_prefer_floor":
        return np.array([ratio <= 0.5, ratio > 0.5])
    if mode == "round_prefer_ceil":
        return np.array([ratio < 0.5, ratio >= 0.5])
    if mode == "floor":
        return np.array([1, 0])
    if mode == "ceil":
        return np.array([0, 1])
    raise ValueError(f"Unexpected value {mode!r}.")



def _interpolate_1d_with_x(
    data: np.ndarray,
    scale_factor: float,
    output_width_int: int,
    x: float,
    get_coeffs: Callable[[float, float], np.ndarray],
    roi: np.ndarray | None = None,
    extrapolation_value: float = 0.0,
    coordinate_transformation_mode: str = "half_pixel",
    exclude_outside: bool = False,
) -> np.ndarray:
    
    input_width = len(data)
    output_width = scale_factor * input_width

    if coordinate_transformation_mode == "align_corners":
        if output_width == 1:
            x_ori = 0.0
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    elif coordinate_transformation_mode == "asymmetric":
        x_ori = x / scale_factor
    elif coordinate_transformation_mode == "tf_crop_and_resize":
        if roi is None:
            raise ValueError("roi cannot be None.")
        if output_width == 1:
            x_ori = (roi[1] - roi[0]) * (input_width - 1) / 2
        else:
            x_ori = x * (roi[1] - roi[0]) * (input_width - 1) / (output_width - 1)
        x_ori += roi[0] * (input_width - 1)

        if x_ori < 0 or x_ori > input_width - 1:
            return np.array(extrapolation_value)
    elif coordinate_transformation_mode == "pytorch_half_pixel":
        if output_width == 1:
            x_ori = -0.5
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel":
        x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel_symmetric":
        adjustment = output_width_int / output_width
        center = input_width / 2
        offset = center * (1 - adjustment)
        x_ori = offset + (x + 0.5) / scale_factor - 0.5
    else:
        raise ValueError(
            f"Invalid coordinate_transformation_mode: {coordinate_transformation_mode!r}."
        )

    x_ori_int = np.floor(x_ori).astype(int).item()

    if x_ori.is_integer():
        ratio = 1
    else:
        ratio = x_ori - x_ori_int

    coeffs = get_coeffs(ratio, scale_factor)
    n = len(coeffs)

    idxes, points = _get_neighbor(x_ori, n, data)

    if exclude_outside:
        for i, idx in enumerate(idxes):
            if idx < 0 or idx >= input_width:
                coeffs[i] = 0
        coeffs /= sum(coeffs)

    return np.dot(coeffs, points).item()  


def _interpolate_nd_with_x(
    data: np.ndarray,
    n: int,
    scale_factors: list[float],
    output_size: list[int],
    x: list[float],
    get_coeffs: Callable[[float, float], np.ndarray],
    roi: np.ndarray | None = None,
    exclude_outside: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    
    if n == 1:
        return _interpolate_1d_with_x(
            data,
            scale_factors[0],
            output_size[0],
            x[0],
            get_coeffs,
            roi=roi,
            exclude_outside=exclude_outside,
            **kwargs,
        )
    res1d = []

    for i in range(data.shape[0]):
        r = _interpolate_nd_with_x(
            data[i],
            n - 1,
            scale_factors[1:],
            output_size[1:],
            x[1:],
            get_coeffs,
            roi=None if roi is None else np.concatenate([roi[1:n], roi[n + 1 :]]),
            exclude_outside=exclude_outside,
            **kwargs,
        )
        res1d.append(r)
    

    return _interpolate_1d_with_x(
        res1d, 
        scale_factors[0],
        output_size[0],
        x[0],
        get_coeffs,
        roi=None if roi is None else [roi[0], roi[n]], 
        exclude_outside=exclude_outside,
        **kwargs,
    )


def _get_all_coords(data: np.ndarray) -> np.ndarray:
    return _cartesian(
        [list(range(data.shape[i])) for i in range(len(data.shape))]  
    )


def interpolate_nd(
    data: np.ndarray,
    get_coeffs: Callable[[float, float], np.ndarray],
    output_size: list[int] | None = None,
    scale_factors: list[float] | None = None,
    axes: list[int] | None = None,
    roi: np.ndarray | None = None,
    keep_aspect_ratio_policy: str | None = "stretch",
    exclude_outside: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    if output_size is None and scale_factors is None:
        raise ValueError("output_size is None and scale_factors is None.")

    r = len(data.shape)
    if axes is not None:
        if scale_factors is not None:
            new_scale_factors = [1.0] * r
            for i, d in enumerate(axes):
                new_scale_factors[d] = scale_factors[i]
            scale_factors = new_scale_factors

        if output_size is not None:
            new_output_size = [data.shape[i] for i in range(r)]
            for i, d in enumerate(axes):
                new_output_size[d] = output_size[i]
            output_size = new_output_size


        if roi is not None:
            new_roi = ([0.0] * r) + ([1.0] * r)
            naxes = len(axes)
            for i, d in enumerate(axes):
                new_roi[d] = roi[i]
                new_roi[r + d] = roi[naxes + i]
            roi = new_roi 
    else:
        axes = list(range(r))

    if output_size is not None:
        scale_factors = [output_size[i] / data.shape[i] for i in range(r)]
        if keep_aspect_ratio_policy != "stretch":
            if keep_aspect_ratio_policy == "not_larger":
                scale = np.array(scale_factors)[axes].min()
            elif keep_aspect_ratio_policy == "not_smaller":
                scale = np.array(scale_factors)[axes].max()
            else:
                raise ValueError(
                    f"Invalid keep_aspect_ratio_policy={keep_aspect_ratio_policy!r}"
                )

            scale_factors = [scale if i in axes else 1.0 for i in range(r)]

            def round_half_up(x: float) -> int:
                return int(x + 0.5)

            output_size = [
                round_half_up(scale * data.shape[i]) if i in axes else data.shape[i]
                for i in range(r)
            ]

    else:
        output_size = (scale_factors * np.array(data.shape)).astype(int) 
    if scale_factors is None:
        raise ValueError("scale_factors is None.")
    if output_size is None:
        raise ValueError("output_size is None.")

    ret = np.zeros(output_size)
    for x in _get_all_coords(ret):
        ret[tuple(x)] = _interpolate_nd_with_x(
            data,
            len(data.shape),
            scale_factors,
            output_size,
            x,
            get_coeffs,
            roi=roi,
            exclude_outside=exclude_outside,
            **kwargs,
        )
    return ret


def cubic_coeffs(
    ratio: float, scale: float | None = None, A: float = -0.75
) -> np.ndarray:
    del scale  # Unused
    coeffs = [
        ((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A,
        ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
        ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
        ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A)
        * ((1 - ratio) + 1)
        - 4 * A,
    ]
    return np.array(coeffs)


    

class Resize(RunAll):

    @staticmethod
    def resize_upsample_scales_nearest() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), scale_factors=scales
        ).astype(np.float32)

        x = [data, scales]
        y = output
        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_nearest"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::HALF_PIXEL_SYMMETRIC),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_downsample_scales_nearest() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), scale_factors=scales
        ).astype(np.float32)

        x = [data, scales]
        y = output
        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_nearest"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_upsample_sizes_nearest() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 7, 8], dtype=np.int64)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), output_size=sizes
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
           
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_sizes_nearest"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_downsample_sizes_nearest() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 1, 3], dtype=np.int64)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), output_size=sizes
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_sizes_nearest"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_upsample_scales_linear() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = interpolate_nd(
            data, lambda x, _: linear_coeffs(x, None), scale_factors=scales
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_linear"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

        
    @staticmethod
    def resize_upsample_scales_linear_align_corners() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x, None),
            scale_factors=scales,
            coordinate_transformation_mode="align_corners",
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_linear_align_corners"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::ALIGN_CORNERS)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_downsample_scales_linear() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = interpolate_nd(
            data, lambda x, _: linear_coeffs(x, None), scale_factors=scales
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_linear"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_downsample_scales_linear_align_corners() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x, None),
            scale_factors=scales,
            coordinate_transformation_mode="align_corners",
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_linear_align_corners"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::ALIGN_CORNERS)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_upsample_scales_cubic() -> None:

        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = interpolate_nd(
            data, lambda x, _: cubic_coeffs(x, None), scale_factors=scales
        ).astype(np.float32)
        
        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_cubic"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_upsample_scales_cubic_align_corners() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = interpolate_nd(
            data,
            lambda x, _: cubic_coeffs(x),
            scale_factors=scales,
            coordinate_transformation_mode="align_corners",
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_cubic_align_corners"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::ALIGN_CORNERS)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_downsample_scales_cubic() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
        output = interpolate_nd(
            data, lambda x, _: cubic_coeffs(x), scale_factors=scales
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_cubic"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_downsample_scales_cubic_align_corners() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

        output = interpolate_nd(
            data,
            lambda x, _: cubic_coeffs(x),
            scale_factors=scales,
            coordinate_transformation_mode="align_corners",
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_cubic_align_corners"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::ALIGN_CORNERS)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

        
    @staticmethod
    def resize_upsample_sizes_cubic() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 9, 10], dtype=np.int64)
        output = interpolate_nd(
            data, lambda x, _: cubic_coeffs(x), output_size=sizes
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_sizes_cubic"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

        
    @staticmethod
    def resize_downsample_sizes_cubic() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 3, 3], dtype=np.int64)

        output = interpolate_nd(
            data, lambda x, _: cubic_coeffs(x), output_size=sizes
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_sizes_cubic"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)





    @staticmethod
    def resize_upsample_scales_cubic_A_n0p5_exclude_outside() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        output = interpolate_nd(
            data,
            lambda x, _: cubic_coeffs(x, A=-0.5),
            scale_factors=scales,
            exclude_outside=True,
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_cubic_A_n0p5_exclude_outside"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::Some(FixedTrait::<FP16x16>::new(32768, true)),"
        func_sig += "Option::Some(true),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_downsample_scales_cubic_A_n0p5_exclude_outside() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
        output = interpolate_nd(
            data,
            lambda x, _: cubic_coeffs(x, A=-0.5),
            scale_factors=scales,
            exclude_outside=True,
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_cubic_A_n0p5_exclude_outside"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::Some(FixedTrait::<FP16x16>::new(32768, true)),"
        func_sig += "Option::Some(true),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)



    @staticmethod
    def resize_upsample_scales_cubic_asymmetric() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = interpolate_nd(
            data,
            lambda x, _: cubic_coeffs(x, A=-0.75),
            scale_factors=scales,
            coordinate_transformation_mode="asymmetric",
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_cubic_asymmetric"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::ASYMMETRIC)," 
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)



    @staticmethod
    def resize_tf_crop_and_resize() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
        sizes = np.array([1, 1, 3, 3], dtype=np.int64)

        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x),
            output_size=sizes,
            roi=roi,
            coordinate_transformation_mode="tf_crop_and_resize",
        ).astype(np.float32)
        x = [data, sizes, roi]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.FP16x16, x[2].shape, to_fp(x[2].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_tf_crop_and_resize"
        func_sig = "data.resize("
        func_sig += "roi,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::TF_CROP_AND_RESIZE)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)

    @staticmethod
    def resize_tf_crop_and_resize_extrapolation_value() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
        sizes = np.array([1, 1, 3, 3], dtype=np.int64)
        
        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x),
            output_size=sizes,
            roi=roi,
            coordinate_transformation_mode="tf_crop_and_resize",
            extrapolation_value=10.0,
        ).astype(np.float32)

        x = [data, sizes, roi]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.FP16x16, x[2].shape, to_fp(x[2].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_tf_crop_and_resize_extrapolation_value"
        func_sig = "data.resize("
        func_sig += "roi,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::TF_CROP_AND_RESIZE)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(FixedTrait::<FP16x16>::new(655360, false)),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)

    @staticmethod
    def resize_downsample_sizes_linear_pytorch_half_pixel() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 3, 1], dtype=np.int64)
        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x),
            output_size=sizes,
            coordinate_transformation_mode="pytorch_half_pixel",
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten())  
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_downsample_sizes_linear_pytorch_half_pixel"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::PYTORCH_HALF_PIXEL)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)



    @staticmethod
    def resize_upsample_sizes_nearest_floor_align_corners() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 8, 8], dtype=np.int64)
        output = interpolate_nd(
            data,
            lambda x, _: nearest_coeffs(x, mode="floor"),
            output_size=sizes,
            coordinate_transformation_mode="align_corners",
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten())  
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_upsample_sizes_nearest_floor_align_corners"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::ALIGN_CORNERS)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::Some(NEAREST_MODE::FLOOR),)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 8, 8], dtype=np.int64)

        output = interpolate_nd(
            data,
            lambda x, _: nearest_coeffs(x, mode="round_prefer_ceil"),
            output_size=sizes,
            coordinate_transformation_mode="asymmetric",
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten())  
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::ASYMMETRIC)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::Some(NEAREST_MODE::ROUND_PREFER_CEIL),)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_upsample_sizes_nearest_ceil_half_pixel() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 8, 8], dtype=np.int64)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x, mode="ceil"), output_size=sizes
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten())  
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_upsample_sizes_nearest_ceil_half_pixel"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::HALF_PIXEL)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::Some(NEAREST_MODE::CEIL),)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_downsample_scales_linear_antialias() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = interpolate_nd(
            data, linear_coeffs_antialias, scale_factors=scales
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_linear_antialias"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(1),"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_downsample_sizes_linear_antialias() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 3, 3], dtype=np.int64)

        output = interpolate_nd(
            data, linear_coeffs_antialias, output_size=sizes
        ).astype(np.float32)

        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten())  
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_downsample_sizes_linear_pytorch_half_pixel"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::Some(1),"
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)


    @staticmethod
    def resize_downsample_scales_cubic_antialias() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = interpolate_nd(
            data, cubic_coeffs_antialias, scale_factors=scales
        ).astype(np.float32)

        x = [data, scales]
        y = output

        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_cubic_antialias"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(1),"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_downsample_sizes_cubic_antialias() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 1, 3, 3], dtype=np.int64)

        output = interpolate_nd(data, cubic_coeffs_antialias, output_size=sizes).astype(
            np.float32
        )
        x = [data, sizes]
        y = output
        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten())  
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        name = "resize_downsample_sizes_cubic_antialias"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "sizes,"
        func_sig += "Option::Some(1),"
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::CUBIC),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)

    @staticmethod
    def resize_upsample_scales_nearest_axes_2_3() -> None:
        axes = np.array([2, 3], dtype=np.int64)
        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([2.0, 3.0], dtype=np.float32)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), scale_factors=scales, axes=axes
        ).astype(np.float32)

        x = [data, scales, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.FP16x16, x[1].shape, to_fp(x[1].flatten(), FixedImpl.FP16x16)) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_nearest_axes_2_3"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)

    @staticmethod
    def resize_upsample_scales_nearest_axes_3_2() -> None:
        
        axes = np.array([3, 2], dtype=np.int64)
        data = np.array([[[[1, 2],[3, 4],]]],dtype=np.float32,)

        scales = np.array([3.0, 2.0], dtype=np.float32)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), scale_factors=scales, axes=axes
        ).astype(np.float32)
        x = [data, scales, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.FP16x16, x[1].shape, to_fp(x[1].flatten(), FixedImpl.FP16x16)) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_nearest_axes_3_2"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)

    @staticmethod
    def resize_upsample_sizes_nearest_axes_2_3() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([7, 8], dtype=np.int64)
        axes = np.array([2, 3], dtype=np.int64)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), output_size=sizes, axes=axes
        ).astype(np.float32)

        x = [data, sizes, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_sizes_nearest_axes_2_3"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)

    @staticmethod
    def resize_upsample_sizes_nearest_axes_3_2() -> None:
        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([8, 7], dtype=np.int64)
        axes = np.array([3, 2], dtype=np.int64)

        output = interpolate_nd(
            data, lambda x, _: nearest_coeffs(x), output_size=sizes, axes=axes
        ).astype(np.float32)

        x = [data, sizes, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_sizes_nearest_axes_3_2"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)

        
    @staticmethod
    def resize_tf_crop_and_resize_axes_2_3() -> None:
        axes = np.array([2, 3], dtype=np.int64)
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        roi = np.array([0.4, 0.6, 0.6, 0.8], dtype=np.float32)
        sizes = np.array([3, 3], dtype=np.int64)

        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x),
            output_size=sizes,
            roi=roi,
            axes=axes,
            coordinate_transformation_mode="tf_crop_and_resize",
        ).astype(np.float32)

        x = [data, sizes, roi, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.FP16x16, x[2].shape, to_fp(x[2].flatten(), FixedImpl.FP16x16)) 
        x[3] = Tensor(Dtype.U32, x[3].shape, x[3].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_tf_crop_and_resize_axes_2_3"
        func_sig = "data.resize("
        func_sig += "roi,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::TF_CROP_AND_RESIZE)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2], x[3]], y, func_sig, name)

    @staticmethod
    def resize_tf_crop_and_resize_axes_3_2() -> None:
        axes = np.array([3, 2], dtype=np.int64)
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        roi = np.array([0.6, 0.4, 0.8, 0.6], dtype=np.float32)
        sizes = np.array([3, 3], dtype=np.int64)

        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x),
            output_size=sizes,
            roi=roi,
            axes=axes,
            coordinate_transformation_mode="tf_crop_and_resize",
        ).astype(np.float32)

        x = [data, sizes, roi, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.FP16x16, x[2].shape, to_fp(x[2].flatten(), FixedImpl.FP16x16)) 
        x[3] = Tensor(Dtype.U32, x[3].shape, x[3].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_tf_crop_and_resize_axes_3_2"
        func_sig = "data.resize("
        func_sig += "roi,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::TF_CROP_AND_RESIZE)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2], x[3]], y, func_sig, name)



    @staticmethod
    def resize_upsample_sizes_nearest_not_larger() -> None:
        keep_aspect_ratio_policy = "not_larger"
        axes = np.array([2, 3], dtype=np.int64)
        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([7, 8], dtype=np.int64) 
        output = interpolate_nd(
            data,
            lambda x, _: nearest_coeffs(x),
            output_size=sizes,
            axes=axes,
            keep_aspect_ratio_policy=keep_aspect_ratio_policy,
        ).astype(np.float32)

        x = [data, sizes, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_sizes_nearest_not_larger"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(KEEP_ASPECT_RATIO_POLICY::NOT_LARGER),"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)



    @staticmethod
    def resize_upsample_sizes_nearest_not_smaller() -> None:
        keep_aspect_ratio_policy = "not_smaller"
        axes = np.array([2, 3], dtype=np.int64)
        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([7, 8], dtype=np.int64)  # Results in 8x8

        output = interpolate_nd(
            data,
            lambda x, _: nearest_coeffs(x),
            output_size=sizes,
            axes=axes,
            keep_aspect_ratio_policy=keep_aspect_ratio_policy,
        ).astype(np.float32)

        x = [data, sizes, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_sizes_nearest_not_smaller"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(KEEP_ASPECT_RATIO_POLICY::NOT_SMALLER),"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)




    @staticmethod
    def resize_downsample_sizes_nearest_not_larger() -> None:
        keep_aspect_ratio_policy = "not_larger" 
        axes = np.array([2, 3], dtype=np.int64)
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 3], dtype=np.int64)

        output = interpolate_nd(
            data,
            lambda x, _: nearest_coeffs(x),
            output_size=sizes,
            axes=axes,
            keep_aspect_ratio_policy=keep_aspect_ratio_policy,
        ).astype(np.float32)

        x = [data, sizes, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_sizes_nearest_not_larger"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(KEEP_ASPECT_RATIO_POLICY::NOT_LARGER),"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)



    @staticmethod
    def resize_downsample_sizes_nearest_not_smaller() -> None:
        keep_aspect_ratio_policy = "not_smaller"
        axes = np.array([2, 3], dtype=np.int64)
        
        data = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        sizes = np.array([1, 3], dtype=np.int64)  
        output = interpolate_nd(
            data,
            lambda x, _: nearest_coeffs(x),
            output_size=sizes,
            axes=axes,
            keep_aspect_ratio_policy=keep_aspect_ratio_policy,
        ).astype(np.float32)

        x = [data, sizes, axes]
        y = output

        x[0] = Tensor(Dtype.FP16x16, x[0].shape, to_fp(x[0].flatten(), FixedImpl.FP16x16)) 
        x[1] = Tensor(Dtype.U32, x[1].shape, x[1].flatten()) 
        x[2] = Tensor(Dtype.U32, x[2].shape, x[2].flatten())  

        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_sizes_nearest_not_smaller"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "sizes,"
        func_sig += "Option::None,"
        func_sig += "axes," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(KEEP_ASPECT_RATIO_POLICY::NOT_SMALLER),"
        func_sig += "Option::Some(MODE::NEAREST),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1], x[2]], y, func_sig, name)




    @staticmethod
    def resize_downsample_scales_linear_half_pixel_symmetric() -> None:
        data = np.array([[[[1, 2, 3, 4]]]], dtype=np.float32)
        scales = np.array([1.0, 1.0, 1.0, 0.6], dtype=np.float32)


        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x),
            scale_factors=scales,
            coordinate_transformation_mode="half_pixel_symmetric",
        ).astype(np.float32)

        x = [data, scales]
        y = output
        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_downsample_scales_linear_half_pixel_symmetric"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::HALF_PIXEL_SYMMETRIC),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)



    @staticmethod
    def resize_upsample_scales_linear_half_pixel_symmetric() -> None:
        data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        scales = np.array([1.0, 1.0, 2.3, 2.94], dtype=np.float32)

        output = interpolate_nd(
            data,
            lambda x, _: linear_coeffs(x),
            scale_factors=scales,
            coordinate_transformation_mode="half_pixel_symmetric",
        ).astype(np.float32)

        x = [data, scales]
        y = output
        for i in range(len(x)):
            x[i] = Tensor(Dtype.FP16x16, x[i].shape, to_fp(x[i].flatten(), FixedImpl.FP16x16)) 
        
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "resize_upsample_scales_linear_half_pixel_symmetric"
        func_sig = "data.resize("
        func_sig += "Option::None,"
        func_sig += "scales,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(TRANSFORMATION_MODE::HALF_PIXEL_SYMMETRIC),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(MODE::LINEAR),"
        func_sig += "Option::None,)"
        make_test([x[0], x[1]], y, func_sig, name)



































