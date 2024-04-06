import numpy as np
from nodegen.node import RunAll
from ..helpers import  make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def hardswish(x: np.ndarray) -> np.ndarray:
    alfa = float(1 / 6)
    beta = 0.5
    return x * np.maximum(0, np.minimum(1, alfa * x + beta))

class Hard_swish(RunAll):
    @staticmethod
    def fp8x23():
        def fp8x23_4D():
            x = np.random.uniform(-3, 3, (3, 2, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
    
            name = "hard_swish_fp8x23_4D"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)
            
        def fp8x23_4D_zero_vals():
            x = np.random.uniform(0, 0, (3, 2, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
    
            name = "hard_swish_fp8x23_4D_zero_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)

        def fp8x23_4D_neg_vals():
            x = np.random.uniform(-3, -6, (3, 2, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
    
            name = "hard_swish_fp8x23_4D_neg_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)
            
        def fp8x23_3D():
            x = np.random.uniform(-3, 3, (3, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
    
            name = "hard_swish_fp8x23_3D"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)

        def fp8x23_3D_zero_vals():
            x = np.random.uniform(0, 0, (3, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
    
            name = "hard_swish_fp8x23_3D_zero_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)
        
        def fp8x23_3D_neg_vals():
            x = np.random.uniform(-3, -6, (3, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
    
            name = "hard_swish_fp8x23_3D_neg_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)

        def fp8x23_2D():
            x = np.random.uniform(-3, 3, (3, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
    
            name = "hard_swish_fp8x23_2D"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)

        
        fp8x23_2D()
        fp8x23_3D()
        fp8x23_4D()
        fp8x23_4D_zero_vals()
        fp8x23_4D_neg_vals()
        fp8x23_3D()
        fp8x23_3D_zero_vals()
        fp8x23_3D_neg_vals()

    @staticmethod
    def fp16x16():
        def fp16x16_4D():
            x = np.random.uniform(-3, 3, (3, 2, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32) 
    
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp( y.flatten(), FixedImpl.FP16x16))
    
            name = "hard_swish_fp16x16_4D"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)

        def fp16x16_4D_zero_vals():
            x = np.random.uniform(0, 0, (3, 2, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32) 
    
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp( y.flatten(), FixedImpl.FP16x16))
    
            name = "hard_swish_fp16x16_4D_zero_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)

        def fp16x16_4D_neg_vals():
            x = np.random.uniform(-3, -6, (3, 2, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32) 
    
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp( y.flatten(), FixedImpl.FP16x16))
    
            name = "hard_swish_fp16x16_4D_neg_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name,)

        def fp16x16_3D():
            x = np.random.uniform(-3, 3, (3, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp( y.flatten(), FixedImpl.FP16x16))
    
            name = "hard_swish_fp16x16_3D"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name, )

        def fp16x16_3D_zero_vals():
            x = np.random.uniform(0, 0, (3, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp( y.flatten(), FixedImpl.FP16x16))
    
            name = "hard_swish_fp16x16_3D_zero_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name, )

        def fp16x16_3D_neg_vals():
            x = np.random.uniform(-3, -6, (3, 2, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32)
    
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp( y.flatten(), FixedImpl.FP16x16))
    
            name = "hard_swish_fp16x16_3D_neg_vals"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name, )
            
        def fp16x16_2D():
            x = np.random.uniform(-3, 3, (3, 2)).astype(np.float32)
            y = hardswish(x).astype(np.float32) 
    
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp( y.flatten(), FixedImpl.FP16x16))
    
            name = "hard_swish_fp16x16_2D"
            make_test([_x], _y, "NNTrait::hard_swish(@input_0)", name)

        fp16x16_2D()
        fp16x16_3D()
        fp16x16_4D()
        fp16x16_4D_zero_vals()
        fp16x16_4D_neg_vals()
        fp16x16_3D()
        fp16x16_3D_zero_vals()
        fp16x16_3D_neg_vals()

