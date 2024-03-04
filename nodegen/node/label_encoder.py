import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221
def labelEncoder(  # type: ignore
    x,
    default_float=None,
    default_int64=None,
    default_string=None,
    keys_floats=None,
    keys_int64s=None,
    keys_strings=None,
    values_floats=None,
    values_int64s=None,
    values_strings=None,
):
    keys = keys_floats if keys_floats is not None else (keys_int64s if np.any(keys_int64s) else keys_strings)
    values = values_floats if values_floats is not None else (values_int64s if np.any(values_int64s) else values_strings)

    classes = dict(zip(keys, values))
    if id(keys) == id(keys_floats):
        cast = float
    elif id(keys) == id(keys_int64s):
        cast = int  # type: ignore
    else:
        cast = str  # type: ignore
    if id(values) == id(values_floats):
        defval = default_float
        dtype = np.float32
    elif id(values) == id(values_int64s):
        defval = default_int64
        dtype = np.int64  # type: ignore
    else:
        defval = default_string
        if not isinstance(defval, str):
            defval = ""
        dtype = np.str_  # type: ignore
    shape = x.shape
    if len(x.shape) > 1:
        x = x.flatten()
    res = []
    for i in range(0, x.shape[0]):
        v = classes.get(cast(x[i]), defval)
        res.append(v)
    return np.array(res, dtype=dtype).reshape(shape)

class Label_encoder(RunAll):

    @staticmethod
    def label_encoder_fp16x16():
            
        def labelencoder():
            def default():
                x = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3]).astype(np.int64)
                keys = np.array([1, 2, 5, 6, ]).astype(np.int64)
                values = np.array([11, 22, 55, 66]).astype(np.int64)
                default = np.array(99).astype(np.int64)

                y = labelEncoder(x=x, keys_int64s=keys, values_int64s=values, default_int64=default)

                x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
                default = Tensor(Dtype.FP16x16, default.shape, to_fp(default.flatten(), FixedImpl.FP16x16))
                keys = Tensor(Dtype.FP16x16, keys.shape, to_fp(keys.flatten(), FixedImpl.FP16x16))
                values = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16)) 

                name = "label_encoder_fp16x16_3d_default"
                make_test(
                    inputs = [x, default, keys, values], output = y, func_sig = """input_0.label_encoder(default_list:Option::None, default_tensor: Option::Some(input_1), 
                    keys:Option::None, keys_tensor: Option::Some(input_2),
                    values: Option::None, values_tensor: Option::Some(input_3))""", 
                    name= name)
            
            default()
        labelencoder()

    @staticmethod
    def label_encoder_fp8x23():
            
        def label_encoder():
            def default():
                
                x = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8]).astype(np.int64)
                keys = np.array([1, 2, 5, 6, 7]).astype(np.int64)
                values = np.array([11, 22, 55, 66, 77]).astype(np.int64)
                default = np.array(99).astype(np.int64)

                y = labelEncoder(x=x, keys_int64s=keys, values_int64s=values, default_int64=default)

                x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
                default = Tensor(Dtype.FP8x23, default.shape, to_fp(default.flatten(), FixedImpl.FP8x23))
                keys = Tensor(Dtype.FP8x23, keys.shape, to_fp(keys.flatten(), FixedImpl.FP8x23))
                values = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23)) 

                name = "label_encoder_fp8x23_default"

                make_test(
                    inputs = [x, default, keys, values], output = y, func_sig = """input_0.label_encoder(default_list:Option::None, default_tensor: Option::Some(input_1), 
                    keys:Option::None, keys_tensor: Option::Some(input_2),
                    values: Option::None, values_tensor: Option::Some(input_3))""", 
                    name= name)

          
                
            default()
        label_encoder()

    @staticmethod
    def label_encoder_i8():
            
        def label_encoder_3D():
            def default():

                x = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8]).astype(np.int8)
                keys = np.array([1, 2, 5, 6, 7]).astype(np.int8)
                values = np.array([11, 22, 55, 66, 77]).astype(np.int8)
                default = np.array(99).astype(np.int8)

                y = labelEncoder(x=x, keys_int64s=keys, values_int64s=values, default_int64=default)

                x =  Tensor(Dtype.I8, x.shape, x.flatten())
                default =  Tensor(Dtype.I8, default.shape, default.flatten())
                keys =  Tensor(Dtype.I8, keys.shape, keys.flatten())
                values =  Tensor(Dtype.I8, values.shape, values.flatten())
                y =  Tensor(Dtype.I8, y.shape, y.flatten())

                name = "label_encoder_i8_default"
                make_test(
                    inputs = [x, default, keys, values], output = y, func_sig = """input_0.label_encoder(default_list:Option::None, default_tensor: Option::Some(input_1), 
                    keys:Option::None, keys_tensor: Option::Some(input_2),
                    values: Option::None, values_tensor: Option::Some(input_3))""", 
                    name= name)
                
           
            default()
        label_encoder_3D()


    @staticmethod
    def label_encoder_i32():
        def label_encoder_3D():
            def default():
                x = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8]).astype(np.int32)
                keys = np.array([1, 2, 5, 6, 7]).astype(np.int32)
                values = np.array([11, 22, 55, 66, 77]).astype(np.int32)
                default = np.array(99).astype(np.int32)

                y = labelEncoder(x=x, keys_int64s=keys, values_int64s=values, default_int64=default)

                x =  Tensor(Dtype.I32, x.shape, x.flatten())
                default =  Tensor(Dtype.I32, default.shape, default.flatten())
                keys =  Tensor(Dtype.I32, keys.shape, keys.flatten())
                values =  Tensor(Dtype.I32, values.shape, values.flatten())
                y =  Tensor(Dtype.I32, y.shape, y.flatten())

                name = "label_encoder_i32_default"
                make_test(
                    inputs = [x, default, keys, values], output = y, func_sig = """input_0.label_encoder(default_list:Option::None, default_tensor: Option::Some(input_1), 
                    keys:Option::None, keys_tensor: Option::Some(input_2),
                    values: Option::None, values_tensor: Option::Some(input_3))""", 
                    name= name)
                
                
           
                
            default()
        label_encoder_3D()


    @staticmethod
    def label_encoder_u32():
            
        def label_encoder_3D():
            def default():

                x = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8]).astype(np.uint32)
                keys = np.array([1, 2, 5, 6, 7]).astype(np.uint32)
                values = np.array([11, 22, 55, 66, 77]).astype(np.uint32)
                default = np.array(99).astype(np.uint32)

                y = labelEncoder(x=x, keys_int64s=keys, values_int64s=values, default_int64=default)

                x =  Tensor(Dtype.U32, x.shape, x.flatten())
                default =  Tensor(Dtype.U32, default.shape, default.flatten())
                keys =  Tensor(Dtype.U32, keys.shape, keys.flatten())
                values =  Tensor(Dtype.U32, values.shape, values.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten())
                
                name = "label_encoder_u32_default"

                make_test(
                    inputs = [x, default, keys, values], output = y, func_sig = """input_0.label_encoder(default_list:Option::None, default_tensor: Option::Some(input_1), 
                    keys:Option::None, keys_tensor: Option::Some(input_2),
                    values: Option::None, values_tensor: Option::Some(input_3))""", 
                    name= name)
                
           
            default()
        label_encoder_3D()
