import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl

class Reduce_max(RunAll):
    @staticmethod
    def reduce_max_u32():
        def reduce_max_1D():
            x = np.array([0, 1, 2, 3]).astype(np.uint32)
            y = np.maximum.reduce(x, axis=None, keepdims=True).astype(np.uint32)
            
            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            
            name = "reduce_max_u32_1D"
            make_test([x], y, "input_0.reduce_max(Option::None(())), Option::None(()), Option::None(())", name)
            
        def reduce_max_2D():
            def default():
                x = np.array([0, 1, 2, 3]).astype(np.uint32).reshape(2, 2)
                y = np.maximum.reduce(x, axis=None, keepdims=True).astype(np.uint32)
                
                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                
                name = "reduce_max_u32_2D_default"
                make_test([x], y, "input_0.reduce_max(Option::None(())), Option::None(()), Option::None(())", name)
                
            def keepdims():
                x = np.array([0, 1, 2, 3]).astype(np.uint32).reshape(2, 2)
                y = np.maximum.reduce(x, axis=None, keepdims=False).astype(np.uint32)
                
                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                
                name = "reduce_max_u32_2D_keepdims"
                make_test([x], y, "input_0.reduce_max(Option::None(())), Option::Some(false), Option::None(())", name)
                
            def axis_1():
                x = np.array([0, 1, 2, 3]).astype(np.uint32).reshape(2, 2)
                y = np.maximum.reduce(x, axis=(1), keepdims=True).astype(np.uint32)
                
                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                
                name = "reduce_max_u32_2D_axis_1"
                make_test([x], y, "input_0.reduce_max(Option::Some(array![1].span()), Option::None(()), Option::None(())", name)
                
            default()
            keepdims()
            axis_1()
        reduce_max_1D()
        reduce_max_2D()
        
    