import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

class Gather(RunAll):

    @staticmethod
    def gather_fp16x16():
            
        def default():
            x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
            x2 = np.array([[0,1], [2,1], [0, 2]]).astype(np.uint32)
            y = x1.take(x2, axis=0)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_fp16x16_3d_default"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather(indices:input_1, axis:Option::Some(0))", 
                name= name)
            
        def axis1():
            x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
            x2 = np.array([[0,1], [2,1], [0, 2]]).astype(np.int64)
            y = x1.take(x2, axis=1)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_fp16x16_3d_axis1"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather(indices:input_1, axis:Option::Some(1))", 
                name= name)
            
        def axis2():
            x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
            x2 = np.array([[0,1], [2,1], [0, 2]]).astype(np.int64)
            y = x1.take(x2, axis=2)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_fp16x16_3d_axis2"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather(indices:input_1, axis:Option::Some(2))", 
                name= name)
        
        def negative_indices():
            x1 = np.arange(10).astype(np.float32)
            x2 = np.array([0, -9, -10]).astype(np.int64)
            y = np.take(x1, x2, axis=0)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_negative_indices"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather(indices:input_1, axis:Option::Some(0))", 
                name= name)
        
        def negative_axis():
            x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
            x2 = np.array([[0,1], [2,1], [0, 2]]).astype(np.uint32)
            y = x1.take(x2, axis=-1)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_negative_axis"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather(indices:input_1, axis:Option::Some(-1))", 
                name= name)
            
        default()
        axis1()
        axis2()
        negative_indices()
        negative_axis()
