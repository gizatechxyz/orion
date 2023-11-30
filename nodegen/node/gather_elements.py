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
        def gather_elements_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,3)).astype(np.uint32)
                y = gather_elements(x1, x2, axis=0)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "gather_elements_fp16x16_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=3, size=(3,3,3)).astype(np.uint32)
                y = gather_elements(x1, x2, axis=1)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "gather_elements_fp16x16_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=3, size=(3,3,3)).astype(np.uint32)
                y = gather_elements(x1, x2, axis=2)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "gather_elements_fp16x16_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(2))", 
                    name= name)
                
            default()
            axis1()
            axis2()
        gather_elements_3D()


    @staticmethod
    def gather_elements_fp8x23():
        def gather_elements_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,3)).astype(np.int64)
                y = gather_elements(x1, x2, axis=0)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "gather_elements_fp8x23_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=3, size=(3,3,3)).astype(np.int64)
                y = gather_elements(x1, x2, axis=1)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "gather_elements_fp8x23_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=3, size=(3,3,3)).astype(np.int64)
                y = gather_elements(x1, x2, axis=2)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "gather_elements_fp8x23_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(2))", 
                    name= name)
                
            default()
            axis1()
            axis2()
        gather_elements_3D()


    @staticmethod
    def gather_elements_i8():
        def gather_elements_3D():
            def default():
                x1 = np.arange(0,9).reshape(3,3).astype(np.int8)
                x2 = np.random.randint(low = 0,high=2, size=(3,3)).astype(np.int8)
                y = gather_elements(x1, x2, axis=0)

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "gather_elements_i8_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,9).reshape(3,3).astype(np.int8)
                x2 = np.random.randint(low = 0,high=2, size=(3,3)).astype(np.int8)
                y = gather_elements(x1, x2, axis=1)

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "gather_elements_i8_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(1))", 
                    name= name)
                
            default()
            axis1()
        gather_elements_3D()


    @staticmethod
    def gather_elements_i32():
        def gather_elements_3D():
            def default():
                x1 = np.arange(0,24).reshape(4,2,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=2, size=(5,2,3)).astype(np.int32)
                y = gather_elements(x1, x2, axis=0)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "gather_elements_i32_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,24).reshape(4,2,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=2, size=(4,3,3)).astype(np.int32)
                y = gather_elements(x1, x2, axis=1)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "gather_elements_i32_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,24).reshape(4,2,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=2, size=(4,2,4)).astype(np.int32)
                y = gather_elements(x1, x2, axis=2)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "gather_elements_i32_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(2))", 
                    name= name)
                
            default()
            axis1()
            axis2()
        gather_elements_3D()

    @staticmethod
    def gather_elements_u32():
        def gather_elements_3D():
            def default():
                x1 = np.arange(0,108).reshape(3,3,4,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=3, size=(10,3,4,3)).astype(np.int32)
                y = gather_elements(x1, x2, axis=0)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "gather_elements_u32_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,108).reshape(3,3,4,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=3, size=(3,5,4,3)).astype(np.int32)
                y = gather_elements(x1, x2, axis=1)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "gather_elements_u32_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,108).reshape(3,3,4,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=3, size=(3,3,4,3)).astype(np.int32)
                y = gather_elements(x1, x2, axis=2)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "gather_elements_u32_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(2))", 
                    name= name)
                
            def axis3():
                x1 = np.arange(0,108).reshape(3,3,4,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=3, size=(3,3,4,6)).astype(np.int32)
                y = gather_elements(x1, x2, axis=3)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "gather_elements_u32_axis3"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_elements(indices:input_1, axis:Option::Some(3))", 
                    name= name)
                
            default()
            axis1()
            axis2()
            axis3()
        gather_elements_3D()