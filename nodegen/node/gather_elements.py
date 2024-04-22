import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def gather_elements(data, indices, axis=0):  # type: ignore
    data_swaped = np.swapaxes(data, 0, axis)
    index_swaped = np.swapaxes(indices, 0, axis)
    gathered = np.choose(index_swaped, data_swaped, mode="wrap")
    y = np.swapaxes(gathered, 0, axis)
    return y

class Gather_elements(RunAll):

    @staticmethod
    def gather_elements_fp16x16():
        def default():
            x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
            x2 = np.random.randint(low = 0,high=2, size=(3,3,3)).astype(np.uint32)
            y = gather_elements(x1, x2, axis=0)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_elements_default"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(0))", 
                name= name)
            
        def axis1():
            x1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            x2 = np.array([[0, 0], [1, 0]], dtype=np.int32)
            y = gather_elements(x1, x2, axis=1)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_elements_axis1"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(1))", 
                name= name)
            
        def axis2():
            x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
            x2 = np.random.randint(low = 0,high=3, size=(3,3,3)).astype(np.uint32)
            y = gather_elements(x1, x2, axis=2)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_elements_axis2"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(2))", 
                name= name)
        
        def negative_indices():
            x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
            x2 = np.array([[-1, -2, 0], [-2, 0, 0]], dtype=np.int32)
            y = gather_elements(x1, x2, axis=0)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten()) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16)) 

            name = "gather_elements_negative_indices"
            make_test(
                inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(0))", 
                name= name)
            
        default()
        axis1()
        axis2()
        negative_indices()

