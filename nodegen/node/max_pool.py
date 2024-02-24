import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

import numpy as np
import numpy as np

from typing import Tuple, Union
from onnx.reference.ops._op_common_pool import CommonPool


def max_pool( 
    x,
    auto_pad=None,
    ceil_mode=None,
    dilations=None,
    kernel_shape=None,
    pads=None,
    storage_order=None,
    strides=None,
    output_len=None
):
    if (
        dilations is not None
        and (min(dilations) != max(dilations) or min(dilations) != 1)
    ) or (
        strides is not None and (min(strides) != max(strides) or min(strides) != 1)
    ):
        return _max_pool(
            x,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=dilations,
            kernel_shape=kernel_shape,
            pads=pads,
            storage_order=storage_order,
            strides=strides,
            output_len=output_len
        )
    
    return common_pool(        
        "MAX",
        0,
        x,
        auto_pad=auto_pad,
        ceil_mode=ceil_mode,
        dilations=dilations,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        p=1
    )
def _max_pool(  # type: ignore
    
    x,
    auto_pad,
    ceil_mode,
    dilations,
    kernel_shape,
    pads,
    storage_order,
    strides,
    output_len
):
    if pads is None:
        pads = [0 for i in range(len(kernel_shape) * 2)]
    if strides is None:
        strides = [1 for i in range(len(kernel_shape))]
    if dilations is None:
        dilations = [1 for i in range(len(kernel_shape))]
    n_dims = len(kernel_shape)
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    input_spatial_shape = x.shape[2:]
    output_spatial_shape = [0 for s in input_spatial_shape]
    if ceil_mode:
        for i in range(len(input_spatial_shape)):
            output_spatial_shape[i] = int(
                np.ceil(
                    (
                        input_spatial_shape[i]
                        + new_pads[i].sum()
                        - ((kernel_shape[i] - 1) * dilations[i] + 1)
                    )
                    / strides[i]
                    + 1
                )
            )
            need_to_reduce_out_size_in_ceil_mode = (
                output_spatial_shape[i] - 1
            ) * strides[i] >= input_spatial_shape[i] + new_pads[i][0]
            if need_to_reduce_out_size_in_ceil_mode:
                output_spatial_shape[i] -= 1
    else:
        for i in range(len(input_spatial_shape)):
            output_spatial_shape[i] = int(
                np.floor(
                    (
                        input_spatial_shape[i]
                        + new_pads[i].sum()
                        - ((kernel_shape[i] - 1) * dilations[i] + 1)
                    )
                    / strides[i]
                    + 1
                )
            )
    if auto_pad and auto_pad != "NOTSET":
        # Deprecated attribute
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            for i in range(len(input_spatial_shape)):
                if auto_pad == "SAME_UPPER":
                    output_spatial_shape[i] = int(
                        np.ceil(input_spatial_shape[i] / strides[i])
                    )
                else:
                    output_spatial_shape[i] = int(
                        np.floor(input_spatial_shape[i] / strides[i])
                    )
                pad_i = (
                    (output_spatial_shape[i] - 1) * strides[i]
                    + ((kernel_shape[i] - 1) * dilations[i] + 1)
                    - input_spatial_shape[i]
                )
                new_pads[i, 0] = pad_i // 2
                new_pads[i, 1] = pad_i - new_pads[i, 0]
        else:
            for i in range(len(input_spatial_shape)):
                output_spatial_shape[i] = int(
                    np.ceil(
                        (
                            input_spatial_shape[i]
                            - ((kernel_shape[i] - 1) * dilations[i] + 1)
                            + 1
                        )
                        / strides[i]
                    )
                )
    if len(input_spatial_shape) == 1:
        return _max_pool_1d(
            x,
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            new_pads,
            storage_order,
            strides,
            output_spatial_shape,
            output_len
        )
    if len(input_spatial_shape) == 2:
        return _max_pool_2d(
            x,
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            new_pads,
            storage_order,
            strides,
            output_spatial_shape,
            output_len
        )
    if len(input_spatial_shape) == 3:
        return _max_pool_3d(
            x,
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            new_pads,
            storage_order,
            strides,
            output_spatial_shape,
            output_len
        )
    return _max_pool_nd(
        x,
        auto_pad,
        ceil_mode,
        dilations,
        kernel_shape,
        new_pads,
        storage_order,
        strides,
        output_spatial_shape,
        output_len
    )
def _max_pool_1d(  # type: ignore
    
    x,
    auto_pad,
    ceil_mode,
    dilations,
    kernel_shape,
    new_pads,
    storage_order,
    strides,
    output_spatial_shape,
    output_len
):
    global_pooling = False
    y_dims = x.shape[:2] + tuple(output_spatial_shape)
    y = np.zeros(y_dims, dtype=x.dtype)
    indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
    x_dims = x.shape
    channels = x_dims[1]
    height = x_dims[2]
    pooled_height = y_dims[2]
    total_channels = x_dims[0] * channels
    stride_h = 1 if global_pooling else strides[0]
    x_step = height
    y_step = pooled_height
    dilation_h = dilations[0]
    X_data = x.ravel()
    Y_data = y.ravel()
    I_data = indices.ravel()
    def iteration(c):
        x_d = c * x_step
        y_d = c * y_step
        i_d = c * y_step
        for ph in range(pooled_height):
            hstart = ph * stride_h - new_pads[0, 0]
            hend = hstart + kernel_shape[0] * dilation_h
            Yh = None
            h_index = -1
            for h in range(hstart, hend, dilation_h):
                if h < 0 or h >= height:
                    continue
                if Yh is None or X_data[x_d + h] > Yh:
                    Yh = X_data[x_d + h]
                    h_index = h
            Y_data[y_d + ph] = Yh
            I_data[i_d + ph] = c * x_step + h_index
    for c in range(total_channels):
        iteration(c)
    if output_len == 1:  # type: ignore
        return (Y_data.reshape(y_dims),)
    return (Y_data.reshape(y_dims), I_data.reshape(y_dims))
def _max_pool_2d(  # type: ignore
    
    x,
    auto_pad,
    ceil_mode,
    dilations,
    kernel_shape,
    new_pads,
    storage_order,
    strides,
    output_spatial_shape,
    output_len
):
    global_pooling = False
    y_dims = x.shape[:2] + tuple(output_spatial_shape)
    y = np.zeros(y_dims, dtype=x.dtype)
    indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
    x_dims = x.shape
    channels = x_dims[1]
    height = x_dims[2]
    width = x_dims[3] if len(kernel_shape) > 1 else 1
    pooled_height = y_dims[2]
    pooled_width = y_dims[3] if len(kernel_shape) > 1 else 1
    total_channels = x_dims[0] * channels
    stride_h = 1 if global_pooling else strides[0]
    stride_w = 1 if global_pooling else strides[1]
    x_step = height * width
    y_step = pooled_height * pooled_width
    dilation_h = dilations[0]
    dilation_w = dilations[1]
    X_data = x.ravel()
    Y_data = y.ravel()
    I_data = indices.ravel()
    def iteration(c):  # type: ignore
        x_d = c * x_step  # X_data
        y_d = c * y_step  # Y_data
        for ph in range(pooled_height):
            hstart = ph * stride_h - new_pads[0, 0]
            hend = hstart + kernel_shape[0] * dilation_h
            for pw in range(pooled_width):
                wstart = pw * stride_w - new_pads[1, 0]
                wend = wstart + kernel_shape[1] * dilation_w
                
                pool_index = ph * pooled_width + pw
                Yh = None
                h_index = -1
                w_index = -1
                for h in range(hstart, hend, dilation_h):
                    if h < 0 or h >= height:
                        continue
                    for w in range(wstart, wend, dilation_w):
                        if w < 0 or w >= width:
                            continue
                        input_index = h * width + w
                        if input_index < 0 or input_index > X_data.shape[0]:
                            continue
                        if Yh is None or X_data[x_d + input_index] > Yh:
                            Yh = X_data[x_d + input_index]
                            h_index = h
                            w_index = w
                if Yh is None:
                    continue
                Y_data[y_d + pool_index] = Yh
                I_data[y_d + pool_index] = (
                    c * x_step + h_index * width + w_index
                    if storage_order == 0
                    else c * x_step + h_index + w_index * height
                )
    for c in range(total_channels):
        iteration(c)
    if output_len == 1:  # type: ignore
        return (Y_data.reshape(y_dims),)
    return (Y_data.reshape(y_dims), I_data.reshape(y_dims))
def _max_pool_3d(  # type: ignore
    
    x,
    auto_pad,
    ceil_mode,
    dilations,
    kernel_shape,
    new_pads,
    storage_order,
    strides,
    output_spatial_shape,
    output_len
):
    global_pooling = False
    y_dims = x.shape[:2] + tuple(output_spatial_shape)
    y = np.zeros(y_dims, dtype=x.dtype)
    indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
    x_dims = x.shape
    channels = x_dims[1]
    height = x_dims[2]
    width = x_dims[3] if len(kernel_shape) > 1 else 1
    depth = x_dims[4] if len(kernel_shape) > 2 else 1
    pooled_height = y_dims[2]
    pooled_width = y_dims[3] if len(kernel_shape) > 1 else 1
    pooled_depth = y_dims[4] if len(kernel_shape) > 2 else 1
    total_channels = x_dims[0] * channels
    stride_h = 1 if global_pooling else strides[0]
    stride_w = 1 if global_pooling else strides[1]
    stride_d = 1 if global_pooling else strides[2]
    x_step = height * width * depth
    y_step = pooled_height * pooled_width * pooled_depth
    dilation_h = dilations[0]
    dilation_w = dilations[1]
    dilation_d = dilations[2]
    X_data = x.ravel()
    Y_data = y.ravel()
    I_data = indices.ravel()
    def iteration(c):
        x_d = c * x_step
        y_d = c * y_step
        i_d = c * y_step
        for ph in range(pooled_height):
            hstart = ph * stride_h - new_pads[0, 0]
            hend = hstart + kernel_shape[0] * dilation_h
            for pw in range(pooled_width):
                wstart = pw * stride_w - new_pads[1, 0]
                wend = wstart + kernel_shape[1] * dilation_w
                for pd in range(pooled_depth):
                    dstart = pd * stride_d - new_pads[2, 0]
                    dend = dstart + kernel_shape[2] * dilation_d
                    pool_index = (
                        ph * pooled_width * pooled_depth + pw * pooled_depth + pd
                    )
                    Yh = None
                    h_index = -1
                    w_index = -1
                    d_index = -1
                    for h in range(hstart, hend, dilation_h):
                        if h < 0 or h >= height:
                            continue
                        for w in range(wstart, wend, dilation_w):
                            if w < 0 or w >= width:
                                continue
                            for d in range(dstart, dend, dilation_d):
                                if d < 0 or d >= depth:
                                    continue
                                input_index = h * width * depth + w * depth + d
                                if Yh is None or X_data[x_d + input_index] > Yh:
                                    Yh = X_data[x_d + input_index]
                                    h_index = h
                                    w_index = w
                                    d_index = d
                                    

                    Y_data[y_d + pool_index] = Yh
                    I_data[i_d + pool_index] = (
                        (
                            c * x_step
                            + h_index * width * depth
                            + w_index * depth
                            + d_index
                        )
                        if storage_order == 0
                        else (
                            c * x_step
                            + h_index
                            + w_index * height
                            + d_index * height * width
                        )
                    )
    for c in range(total_channels):
        iteration(c)
    if output_len == 1:  # type: ignore
        return (Y_data.reshape(y_dims),)
    return (Y_data.reshape(y_dims), I_data.reshape(y_dims))
def stride(arr):
    stride = np.zeros(len(arr))
    acc = 1
    for i in range(len(arr)):
        stride[i] = acc
        acc *= arr[-(i + 1)]
    return np.flip(stride) 
def reverse_stride(arr):
    stride = np.zeros(len(arr))
    acc = 1
    for i in range(len(arr)):
        acc *= arr[i]
        stride[i] = acc
        
    return stride 


def _max_pool_nd(  # type: ignore
    
    x,
    auto_pad,
    ceil_mode,
    dilations,
    kernel_shape,
    new_pads,
    storage_order,
    strides,
    output_spatial_shape,
    output_len
):
    nd = len(x.shape[2:])
    y_dims = x.shape[:2] + tuple(output_spatial_shape)
    y = np.zeros(y_dims, dtype=x.dtype)
    indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
    x_dims = x.shape
    channels = x_dims[1]   
    x_stride = stride(x.shape)
    y_stride = stride(y_dims)
    total_channels = x_dims[0] * channels
    x_step = x_stride[1]
    y_step = y_stride[1]    
    X_data = x.ravel()
    Y_data = y.ravel()
    I_data = indices.ravel()
    def iteration(c):
        x_d = int(c * x_step)
        y_d = int(c * y_step)
        
        for p in range(int(y_step)):
            pool_index = p
            flatten_index = p
            
            nstart = np.zeros(nd)
            nend = np.zeros(nd)
            nstep = np.zeros(nd)
            
            for n in range(nd):    
                pn, rem = divmod(flatten_index, y_stride[n + 2])
                flatten_index = rem
                
                ns = pn * strides[n] - new_pads[n, 0]
                nstart[n] = ns
                nend[n] = ns + kernel_shape[n] * dilations[n]
                
                nstep[n] = np.ceil((nend[n] - ns) / dilations[n])
                            
            nstride = stride(nstep)
            max_iter = int(nstep[0] * nstride[0])
            n_index = np.full(y_dims, dtype=np.int64, fill_value=-1)
            Yh = None
            
            for i in range(max_iter):
                flatten_index = i
                is_outside = False
                input_index = 0
            
                i_index = np.zeros(nd)

                for n in range(nd):    
                    item, rem = divmod(flatten_index, nstride[n])
                    flatten_index = rem

                    item_ = item * dilations[n] + nstart[n]
                    if item_ < 0 or item_ >= x.shape[2 + n]:
                        is_outside = True
                    i_index[n] = item_
                    input_index += item_ * x_stride[2 + n]
                
                input_index = int(input_index)
                if is_outside == False:
                    if input_index < 0 or input_index > X_data.shape[0]:
                        continue
                    if Yh is None or X_data[x_d + input_index] > Yh:
                        Yh = X_data[x_d + input_index]
                        n_index = i_index
            
            
            Y_data[y_d + p] = Yh
            
    for c in range(total_channels):
        iteration(c)
    if output_len == 1:  # type: ignore
        return (Y_data.reshape(y_dims),)
    return (Y_data.reshape(y_dims), I_data.reshape(y_dims))
                    
                    
                
import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np



def get_pad_shape(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
    output_spatial_shape: Sequence[int],
) -> Sequence[int]:
    spatial_dims = len(input_spatial_shape)
    pad_shape = [0] * spatial_dims
    strides_spatial = strides_spatial or [1] * spatial_dims
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(spatial_dims):
            pad_shape[i] = (
                (output_spatial_shape[i] - 1) * strides_spatial[i]
                + kernel_spatial_shape[i]
                - input_spatial_shape[i]
            )
    elif auto_pad == "VALID":
        pass

    return pad_shape
def get_pad_with_auto_pad(auto_pad: str, pad_shape: Sequence[int]) -> Sequence[int]:
    spatial_dims = len(pad_shape)
    if auto_pad == "SAME_UPPER":
        pads = [pad_shape[i] // 2 for i in range(spatial_dims)] + [
            pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)
        ]
    elif auto_pad == "SAME_LOWER":
        pads = [pad_shape[i] - pad_shape[i] // 2 for i in range(spatial_dims)] + [
            pad_shape[i] // 2 for i in range(spatial_dims)
        ]
    else:
        pads = [0] * spatial_dims * 2  # no padding
    return pads

def get_output_shape_explicit_padding(
    pads: Sequence[int],
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
    dilations: Union[Sequence[int], None] = None,
    ceil_mode: bool = False,
) -> Tuple[Sequence[int], Sequence[int]]:
    
    output_spatial_shape = [0] * len(input_spatial_shape)
    pads = pads or [0] * len(input_spatial_shape) * 2
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    dims = len(input_spatial_shape)
    if dilations is None:
        dilations = np.ones([dims], dtype=np.int64)

    for dim in range(dims):
        dim_size = (
            input_spatial_shape[dim]
            + pads[dim]
            + pads[dims + dim]
            - dilations[dim] * (kernel_spatial_shape[dim] - 1)
            - 1
        ) / strides_spatial[dim] + 1

        if ceil_mode:
            output_spatial_shape[dim] = int(np.ceil(dim_size))
        else:
            output_spatial_shape[dim] = int(np.floor(dim_size))

    pads_spatial_shape_new = pads[:]
    for dim in range(dims):
        sliding_window_size = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
        actual_padded_input_size = (output_spatial_shape[dim] - 1) * strides_spatial[
            dim
        ] + sliding_window_size
        extra_pad = (
            actual_padded_input_size
            - input_spatial_shape[dim]
            - pads[dim]
            - pads[dims + dim]
        )
        if extra_pad > 0:
            pads_spatial_shape_new[dim] += extra_pad // 2
            pads_spatial_shape_new[dims + dim] += extra_pad - extra_pad // 2

    return output_spatial_shape, pads_spatial_shape_new

def get_output_shape_auto_pad(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
) -> Sequence[int]:
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    out_shape = [0] * len(input_spatial_shape)
    for i in range(len(input_spatial_shape)):
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out_shape[i] = (
                math.floor((input_spatial_shape[i] - 1) / strides_spatial[i]) + 1
            )
        elif auto_pad == "VALID":
            out_shape[i] = (
                math.floor(
                    (input_spatial_shape[i] - kernel_spatial_shape[i])
                    / strides_spatial[i]
                )
                + 1
            )
        else:
            raise ValueError(
                "auto_pad can only be NOTSET, SAME_UPPER, SAME_LOWER, or VALID"
            )

    return out_shape

def lp_pool(x: np.array, p: int) -> float:
    y = 0
    for v in np.nditer(x):
        y += abs(v) ** p
    return y ** (1.0 / p)

def pool(
    padded: np.ndarray,
    x_shape: Sequence[int],
    kernel: Sequence[int],
    strides: Sequence[int],
    out_shape: Sequence[int],
    pooling_type: str,
    pads: Union[Sequence[int], None] = None,
    dilations: Union[Sequence[int], None] = None,
    count_include_pad: int = 0,
    p: int = 1,
) -> np.ndarray:
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1], *list(out_shape)], dtype=padded.dtype)
    if dilations is None:
        dilations = np.ones([spatial_size], dtype=np.int64)
    if pads is None:
        pads = np.zeros([spatial_size * 2], dtype=np.int64)
    elif len(pads) == 1:
        pads = pads * spatial_size * 2
    strides = strides or [1] * spatial_size

    def lp_pool_p(x):
        return lp_pool(x, p)



    for shape in itertools.product(
        range(x_shape[0]),
        range(x_shape[1]),
        *[
            range(
                int(
                    (
                        x_shape[i + 2]
                        + pads[i]
                        + pads[i + spatial_size]
                        - (1 + (kernel[i] - 1) * dilations[i])
                    )
                    / strides[i]
                    + 1
                )
            )
            for i in range(spatial_size)
        ],
    ):
        window = padded[shape[0], shape[1]]
        window_vals = np.array(
            [
                window[i]
                for i in list(
                    itertools.product(
                        *[
                            range(
                                strides[i] * shape[i + 2],
                                strides[i] * shape[i + 2]
                                + (1 + (kernel[i] - 1) * dilations[i]),
                                dilations[i],
                            )
                            for i in range(spatial_size)
                        ]
                    )
                )
            ]
        )
        if pooling_type == "AVG":
            f = np.average
        elif pooling_type == "MAX":
            f = np.max
        elif pooling_type == "LPPOOL":
            f = lp_pool_p
        else:
            raise NotImplementedError(
                f"Pooling type {pooling_type} does not support. Should be AVG, MAX"
            )

        if count_include_pad == 1 and (pooling_type in {"AVG", "LPPOOL"}):
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y


def common_pool(
    pooling_type,
    count_include_pad,
    x,
    auto_pad=None,
    ceil_mode=None,
    dilations=None,
    kernel_shape=None,
    pads=None,
    strides=None,
    p=None,
):
    x_shape = np.shape(x)
    pading_value = np.nan if pooling_type == "MAX" or count_include_pad == 0 else 0
    if auto_pad in ["SAME_UPPER", "SAME_LOWER", "VALID"]:
        assert (
            ceil_mode is None or ceil_mode == 0
        ), "ceil_mode is not supported with auto_pad"
        out_shape = get_output_shape_auto_pad(
            auto_pad, x.shape[2:], kernel_shape, strides
        )
        pads_shape = get_pad_shape(
            auto_pad, x_shape[2:], kernel_shape, strides, out_shape
        )
        pads = get_pad_with_auto_pad(auto_pad, pads_shape)
        n_dims = len(pads) // 2
        pads_np = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
        padded = np.pad(
            x,
            ((0, 0), (0, 0), *pads_np),
            mode="constant",
            constant_values=pading_value,
        )
        y = pool(
            padded,
            x_shape,
            kernel_shape,
            strides,
            out_shape,
            pooling_type,
            pads,
            dilations,
            count_include_pad,
            p,
        )
        return (y,)
    else:
        out_shape, pads = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides, dilations, ceil_mode
        )
        # convert pads from [x1_begin, x2_begin,...,x1_end, x2_end,...] to [(x1_begin, x1_end), (x2_begin, x2_end),...]
        n_dims = len(pads) // 2
        pads_np = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
        padded = np.pad(
            x,
            ((0, 0), (0, 0), *pads_np),
            mode="constant",
            constant_values=pading_value,
        )
        y = pool(
            padded,
            x_shape,
            kernel_shape,
            strides,
            out_shape,
            pooling_type,
            pads,
            dilations,
            count_include_pad,
            p,
        )
        return (y,)

class Max_pool(RunAll):
    
    @staticmethod
    def export_maxpool_1d() -> None:
        
        x = np.random.randn(1, 3, 32).astype(np.float32)
        kernel_shape = np.array([2])
        strides = np.array([2])
        padded = x
        y = max_pool(padded, kernel_shape=kernel_shape, strides=strides,output_len=1)
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        

        name = "maxpool_1d"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2].span()),"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_1d_default() -> None:
        
        x = np.random.randn(1, 3, 32).astype(np.float32)
        kernel_shape = np.array([2])
        padded = x
        y = max_pool(padded, kernel_shape=kernel_shape,output_len=1)       
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        

        name = "maxpool_1d_default"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
      
        
    @staticmethod
    def export_maxpool_2d() -> None:
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.float32)
        
        kernel_shape=(2, 2)
        strides=(2, 2)
        padded = x
        y = max_pool(padded,strides = strides,kernel_shape=kernel_shape,output_len=1)
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        

        name = "maxpool_2d"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_2d_default() -> None:
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        kernel_shape = (2, 2)
        padded = x
        y = max_pool(padded, kernel_shape=kernel_shape, output_len=1)
        y = np.array(y[0])
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "maxpool_2d_default"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
      
    def export_maxpool_2d_pads_default() -> None:
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.uint8)
        kernel_shape=(5, 5)
        pads=(2, 2, 2, 2)
        padded = x
        y = max_pool(padded,pads = pads,kernel_shape=kernel_shape,output_len=1)
        y = np.array(y[0])
       

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "maxpool_2d_pads_default"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![5, 5].span(),"
        func_sig += "Option::Some(array![2, 2, 2, 2].span()),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_2d_constraint_index() -> None:
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[7, 9], [17, 19]]]]).astype(np.float32)
        z = np.array([[[[6, 16], [8, 18]]]]).astype(np.int64)

        kernel_shape=(2, 2)
        strides=(2, 2)
        padded = x
        (y, z) = max_pool(padded,strides = strides,kernel_shape=kernel_shape,output_len=2, storage_order=1)
        
        y = np.array(y)
        z = np.array(z)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        z = Tensor(Dtype.I32, z.shape, z.flatten())
        

        name = "maxpool_2d_constraint_index"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(1),"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "1)"
        make_test(
            [x], z, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_2d_same_upper() -> None:
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.float32)

        kernel_shape=(3, 3)
        strides=(2, 2)
        padded = x
        y = max_pool(padded,strides = strides,kernel_shape=kernel_shape,auto_pad="SAME_UPPER")
        y = np.array(y[0])
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        

        name = "maxpool_2d_same_upper"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::Some(AUTO_PAD::SAME_UPPER)," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![3, 3].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_2d_same_upper_default() -> None:
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        kernel_shape = (2, 2)
        padded = x
        y = max_pool(padded,auto_pad="SAME_UPPER", kernel_shape=kernel_shape,  output_len=1)
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "maxpool_2d_same_upper_default"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::Some(AUTO_PAD::SAME_UPPER)," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_2d_same_lower_default() -> None:
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        padded = x
        y = max_pool(padded,auto_pad="SAME_LOWER", kernel_shape=kernel_shape,  output_len=1)
        y = np.array(y[0])
        
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "maxpool_2d_same_lower_default"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::Some(AUTO_PAD::SAME_LOWER)," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_2d_ceil() -> None:
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[11, 12], [15, 16]]]]).astype(np.float32)

        kernel_shape = (3, 3)
        strides = (2, 2)
        padded = x
        y = max_pool(padded,strides = strides, ceil_mode = True,kernel_shape=kernel_shape,  output_len=1)
        y = np.array(y[0])


        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "maxpool_2d_ceil"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(1)," 
        func_sig += "Option::None,"
        func_sig += "array![3, 3].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_2d_dilations() -> None:
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ]
        ).astype(np.float32)

        kernel_shape = (2 , 2)
        dilations = (2, 2)
        padded = x
        y = max_pool(padded,dilations = dilations, ceil_mode = True,kernel_shape=kernel_shape,  output_len=1)
        y = np.array(y[0])        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        

        name = "maxpool_2d_dilations"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_3d_dilations() -> None:
        
        x = np.array(
            [
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                    ]
                ]
            ]
        ).astype(np.float32)
        kernel_shape=(2, 2, 2)
        strides=(1, 1, 1)
        dilations=(2, 2, 2)
        padded = x
        y = max_pool(padded, dilations=dilations, kernel_shape=kernel_shape, strides=strides,output_len=1)
        y = np.array(y[0])
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "maxpool_3d_dilations"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::Some(array![2, 2, 2].span()),"
        func_sig += "array![2, 2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1, 1].span()),"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_maxpool_4d_dilations() -> None:
        x = np.random.randn(1, 3, 4, 4, 4, 4).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2, 2, 2)
        strides = (1, 1, 1, 1)
        dilations = (2, 2, 2, 2)
        padded = x
        y = max_pool(padded,dilations = dilations, ceil_mode = True,kernel_shape=kernel_shape,  output_len=1)
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "maxpool_4d_dilations"
        func_sig = "NNTrait::max_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2, 2, 2].span()),"
        func_sig += "array![2, 2, 2, 2].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "1)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
        
    
    
   