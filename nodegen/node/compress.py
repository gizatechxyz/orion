import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

class Compress(RunAll):

    @staticmethod
    def compress_fp16x16():
            
        def compress_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.array([0, 1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=0)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "compress_fp16x16_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,180).reshape(3,4,3,5).astype(np.int64)
                x2 = np.array([1, 1, 1, 0]).astype(np.int64)
                y = x1.compress(x2, axis=1)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "compress_fp16x16_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,48).reshape(4,3,4).astype(np.int64)
                x2 = np.array([1, 0, 1, 1]).astype(np.int64)
                y = x1.compress(x2, axis=2)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "compress_fp16x16_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(2))", 
                    name= name)
                
            def axis3():
                x1 = np.arange(0,96).reshape(4,3,4, 2).astype(np.int64)
                x2 = np.array([1, 0]).astype(np.int64)
                y = x1.compress(x2, axis=3)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "compress_fp16x16_3d_axis3"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(3))", 
                    name= name)
                
            def noaxis():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1]).astype(np.int64)
                y = x1.compress(x2)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "compress_fp16x16_3d_noaxis"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::None(()))", 
                    name= name)
                
            default()
            axis1()
            axis2()
            axis3()
            noaxis()
        compress_3D()

    @staticmethod
    def compress_fp8x23():
            
        def compress_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.array([0, 1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=0)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "compress_fp8x23_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.array([0, 1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=1)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())  
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23)) 

                name = "compress_fp8x23_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.array([0, 1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=2)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "compress_fp8x23_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(2))", 
                    name= name)
                
            default()
            axis1()
            axis2()
        compress_3D()

    @staticmethod
    def compress_i8():
            
        def compress_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int8)
                x2 = np.array([0, 1, 1]).astype(np.uint8)
                y = x1.compress(x2, axis=0)

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "compress_i8_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int8)
                x2 = np.array([0, 1, 1]).astype(np.uint8)
                y = x1.compress(x2, axis=1)

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten())

                name = "compress_i8_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int8)
                x2 = np.array([0, 1, 1]).astype(np.uint8)
                y = x1.compress(x2, axis=2)

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten())

                name = "compress_i8_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(2))", 
                    name= name)
                
            default()
            axis1()
            axis2()
        compress_3D()

    
    @staticmethod
    def compress_i32():
            
        def compress_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int32)
                x2 = np.array([0, 1, 1]).astype(np.int32)
                y = x1.compress(x2, axis=0)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "compress_i32_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int32)
                x2 = np.array([0, 1, 1]).astype(np.int32)
                y = x1.compress(x2, axis=1)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten())

                name = "compress_i32_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int32)
                x2 = np.array([0, 1, 1]).astype(np.int32)
                y = x1.compress(x2, axis=2)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten())

                name = "compress_i32_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(2))", 
                    name= name)
                
            default()
            axis1()
            axis2()
        compress_3D()

    @staticmethod
    def compress_u32():
            
        def compress_3D():
            def default():
                x1 = np.arange(0,48).reshape(4,4,3).astype(np.uint32)
                x2 = np.array([1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=0)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "compress_u32_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(0))", 
                    name= name)
                
            def axis1():
                x1 = np.arange(0,36).reshape(3,4,3).astype(np.uint32)
                x2 = np.array([0, 1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=1)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.U32, y.shape, y.flatten())

                name = "compress_u32_3d_axis1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(1))", 
                    name= name)
                
            def axis2():
                x1 = np.arange(0,48).reshape(3,4,4).astype(np.uint32)
                x2 = np.array([0, 1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=2)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten())

                name = "compress_u32_3d_axis2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(2))", 
                    name= name)
                
            def axis2_2():
                x1 = np.arange(0,60).reshape(3,4,5).astype(np.uint32)
                x2 = np.array([0, 1, 1]).astype(np.uint32)
                y = x1.compress(x2, axis=2)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten())

                name = "compress_u32_3d_axis2_2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(2))", 
                    name= name)
                
            def axis3():
                x1 = np.arange(0,270).reshape(3,3,5,6).astype(np.uint32)
                x2 = np.array([0, 1, 1,1,0,1]).astype(np.uint32)
                y = x1.compress(x2, axis=3)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten())

                name = "compress_u32_3d_axis3"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.compress(condition:input_1, axis:Option::Some(3))", 
                    name= name)
                
            default()
            axis1()
            axis2()
            axis2_2()
            axis3()
        compress_3D()
