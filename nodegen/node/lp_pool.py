import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

import numpy as np
import numpy as np

from .max_pool import common_pool, get_output_shape_explicit_padding, pool, get_output_shape_auto_pad, get_pad_shape



def lp_pool( 
    x,
    auto_pad=None,
    ceil_mode=None,
    dilations=None,
    kernel_shape=None,
    p=2,
    pads=None,
    strides=None,
    count_include_pad=None,
):
    power_average = common_pool(
        "AVG",
        count_include_pad,
        np.power(np.absolute(x), p),
        auto_pad=auto_pad,
        ceil_mode=ceil_mode,
        dilations=dilations,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )
    kernel_element_count = np.prod(kernel_shape)
    return (np.power(kernel_element_count * power_average[0], 1.0 / p),)


class Lp_pool(RunAll):
    

    @staticmethod
    def export_lppool_1d_default() -> None:
        p = 3
        kernel_shape = [2]
        strides = [1]
        x = np.random.randn(1, 3, 16).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", p=p)
        

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))       

        name = "lppool_1d_default"
        func_sig = "NNTrait::lp_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2].span(),"
        func_sig += "Option::Some(3),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x], y, func_sig, name, Trait.NN)



        
    @staticmethod
    def export_lppool_2d_default() -> None:
        p = 4
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", p=p)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        

        name = "lppool_2d_default"
        func_sig = "NNTrait::lp_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::Some(4),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
        
    @staticmethod
    def export_lppool_3d_default() -> None:
        p = 3
        x = np.random.randn(1, 3, 4, 4, 4).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", p=p)       

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "lppool_3d_default"
        func_sig = "NNTrait::lp_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2, 2].span(),"
        func_sig += "Option::Some(3),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
    
    
    @staticmethod
    def export_lppool_2d_same_upper() -> None:
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape_auto_pad(
            "SAME_UPPER", x_shape[2:], kernel_shape, strides
        )
        pad_shape = get_pad_shape(
            "SAME_UPPER", x_shape[2:], kernel_shape, strides, out_shape
        )
        pad_top = pad_shape[0] // 2
        pad_bottom = pad_shape[0] - pad_top
        pad_left = pad_shape[1] // 2
        pad_right = pad_shape[1] - pad_left
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", pads, p=2)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "lppool_2d_same_upper"
        func_sig = "NNTrait::lp_pool("
        func_sig += "@input_0,"
        func_sig += "Option::Some(AUTO_PAD::SAME_UPPER)," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::Some(2),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1].span()),"
        func_sig += "Option::Some(1))"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_lppool_2d_same_lower() -> None:
        p = 4
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape_auto_pad(
            "SAME_LOWER", x_shape[2:], kernel_shape, strides
        )
        pad_shape = get_pad_shape(
            "SAME_LOWER", x_shape[2:], kernel_shape, strides, out_shape
        )
        pad_bottom = pad_shape[0] // 2
        pad_top = pad_shape[0] - pad_bottom
        pad_right = pad_shape[1] // 2
        pad_left = pad_shape[1] - pad_right
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", pads, p=p)
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "lppool_2d_same_lower"
        func_sig = "NNTrait::lp_pool("
        func_sig += "@input_0,"
        func_sig += "Option::Some(AUTO_PAD::SAME_LOWER)," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::Some(4),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1].span()),"
        func_sig += "Option::Some(1))"
        make_test(
            [x], y, func_sig, name, Trait.NN)

    @staticmethod
    def export_lppool_2d_pads() -> None:
        p = 3
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = pad_top = pad_right = pad_left = 2
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        out_shape, pads = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", pads, p=p)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        name = "lppool_2d_pads"
        func_sig = "NNTrait::lp_pool("
        func_sig += "@input_0,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "array![3, 3].span(),"
        func_sig += "Option::Some(3),"
        func_sig += "Option::Some(array![2, 2, 2, 2].span()),"
        func_sig += "Option::Some(array![1, 1].span()),"
        func_sig += "Option::Some(1))"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
