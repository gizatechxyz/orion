import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
# 687
class Concat(RunAll):
    @staticmethod
    def concat_u32():

        def concat_1D():
            x1 = np.arange(0,3).astype(np.uint32)
            x2 = np.arange(3,6).astype(np.uint32)
            y = np.concatenate((x1, x2))

            x1 = Tensor(Dtype.U32, x1.shape, x1.flatten())
            x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "concat_u32_1d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR)
                    
            
        def concat_2D():
            x1 = np.arange(0,4).astype(np.uint32).reshape(2,2)
            x2 = np.arange(4,8).astype(np.uint32).reshape(2,2)
            y = np.concatenate((x1, x2), axis=0)

            x1 = Tensor(Dtype.U32, x1.shape, x1.flatten())
            x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "concat_u32_2d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR)
            
        def concat_3D():
            def default():
                x1 = np.arange(0,27).astype(np.uint32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.uint32).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=0)

                x1 = Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "concat_u32_3d_default"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_1():
                x1 = np.arange(0,27).astype(np.uint32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.uint32).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=1)

                x1 = Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "concat_u32_3d_axis_1"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_2():
                x1 = np.arange(0,27).astype(np.uint32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.uint32).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=2)

                x1 = Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "concat_u32_3d_axis_2"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def three_tensors_axis_1():
                x1 = np.arange(0,27).astype(np.uint32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.uint32).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.uint32).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=1)

                x1 = Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten())

                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "concat_u32_3d_three_tensors_axis_1"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)
                
            def three_tensors_axis_2():
                x1 = np.arange(0,27).astype(np.uint32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.uint32).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.uint32).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=2)

                x1 = Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten())

                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "concat_u32_3d_three_tensors_axis_2"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            default()
            axis_1()
            axis_2()
            three_tensors_axis_1()
            three_tensors_axis_2()
    
        concat_1D()
        concat_2D()
        concat_3D()

    @staticmethod
    def concat_i32():
        def concat_1D():
            x1 = np.arange(0,3).astype(np.int32)
            x2 = np.arange(3,6).astype(np.int32)
            y = np.concatenate((x1, x2))

            x1 = Tensor(Dtype.I32, x1.shape, x1.flatten())
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "concat_i32_1d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR.TENSOR)
                    
            
        def concat_2D():
            x1 = np.arange(0,4).astype(np.int32).reshape(2,2)
            x2 = np.arange(4,8).astype(np.int32).reshape(2,2)
            y = np.concatenate((x1, x2), axis=0)

            x1 = Tensor(Dtype.I32, x1.shape, x1.flatten())
            x2 = Tensor(Dtype.I32, x2.shape, x2.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "concat_i32_2d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR)
            
        def concat_3D():
            def default():
                x1 = np.arange(0,27).astype(np.int32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int32).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=0)

                x1 = Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.I32, x2.shape, x2.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "concat_i32_3d_default"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_1():
                x1 = np.arange(0,27).astype(np.int32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int32).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=1)

                x1 = Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.I32, x2.shape, x2.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "concat_i32_3d_axis_1"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_2():
                x1 = np.arange(0,27).astype(np.int32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int32).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=2)

                x1 = Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.I32, x2.shape, x2.flatten())
                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "concat_i32_3d_axis_2"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def three_tensors_axis_1():
                x1 = np.arange(0,27).astype(np.int32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int32).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int32).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=1)

                x1 = Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.I32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.I32, x3.shape, x3.flatten())

                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "concat_i32_3d_three_tensors_axis_1"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)
                
            def three_tensors_axis_2():
                x1 = np.arange(0,27).astype(np.int32).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int32).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int32).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=2)

                x1 = Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.I32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.I32, x3.shape, x3.flatten())

                y = Tensor(Dtype.I32, y.shape, y.flatten())

                name = "concat_i32_3d_three_tensors_axis_2"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            default()
            axis_1()
            axis_2()
            three_tensors_axis_1()
            three_tensors_axis_2()

        concat_1D()
        concat_2D()
        concat_3D()
        
    @staticmethod
    def concat_i8():
        def concat_1D():
            x1 = np.arange(0,3).astype(np.int8)
            x2 = np.arange(3,6).astype(np.int8)
            y = np.concatenate((x1, x2))

            x1 = Tensor(Dtype.FP8x23, x1.shape, x1.flatten())
            x2 = Tensor(Dtype.FP8x23, x2.shape, x2.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "concat_i8_1d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR.TENSOR)
                    
            
        def concat_2D():
            x1 = np.arange(0,4).astype(np.int8).reshape(2,2)
            x2 = np.arange(4,8).astype(np.int8).reshape(2,2)
            y = np.concatenate((x1, x2), axis=0)

            x1 = Tensor(Dtype.FP8x23, x1.shape, x1.flatten())
            x2 = Tensor(Dtype.FP8x23, x2.shape, x2.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "concat_i8_2d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR)
            
        def concat_3D():
            def default():
                x1 = np.arange(0,27).astype(np.int8).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int8).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=0)

                x1 = Tensor(Dtype.FP8x23, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.FP8x23, x2.shape, x2.flatten())
                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "concat_i8_3d_default"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_1():
                x1 = np.arange(0,27).astype(np.int8).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int8).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=1)

                x1 = Tensor(Dtype.FP8x23, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.FP8x23, x2.shape, x2.flatten())
                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "concat_i8_3d_axis_1"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_2():
                x1 = np.arange(0,27).astype(np.int8).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int8).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=2)

                x1 = Tensor(Dtype.FP8x23, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.FP8x23, x2.shape, x2.flatten())
                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "concat_i8_3d_axis_2"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def three_tensors_axis_1():
                x1 = np.arange(0,27).astype(np.int8).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int8).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int8).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=1)

                x1 = Tensor(Dtype.FP8x23, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.FP8x23, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.FP8x23, x3.shape, x3.flatten())

                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "concat_i8_3d_three_tensors_axis_1"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)
                
            def three_tensors_axis_2():
                x1 = np.arange(0,27).astype(np.int8).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int8).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int8).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=2)

                x1 = Tensor(Dtype.FP8x23, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.FP8x23, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.FP8x23, x3.shape, x3.flatten())

                y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

                name = "concat_i8_3d_three_tensors_axis_2"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            default()
            axis_1()
            axis_2()
            three_tensors_axis_1()
            three_tensors_axis_2()

        concat_1D()
        concat_2D()
        concat_3D()
        
    @staticmethod
    def concat_fp8x23():
        def concat_1D():
            x1 = np.arange(0,3).astype(np.int64)
            x2 = np.arange(3,6).astype(np.int64)
            y = np.concatenate((x1, x2))

            x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP8x23))  
            x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP8x23)) 
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

            name = "concat_fp8x23_1d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR.TENSOR)
                    
            
        def concat_2D():
            x1 = np.arange(0,4).astype(np.int64).reshape(2,2)
            x2 = np.arange(4,8).astype(np.int64).reshape(2,2)
            y = np.concatenate((x1, x2), axis=0)

            x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP8x23))
            x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP8x23)) 
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

            name = "concat_fp8x23_2d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR)
            
        def concat_3D():
            def default():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=0)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP8x23)) 
                y = Tensor(Dtype.FP8x23, y.shape,to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "concat_fp8x23_3d_default"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_1():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=1)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP8x23)) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "concat_fp8x23_3d_axis_1"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_2():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=2)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP8x23)) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "concat_fp8x23_3d_axis_2"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def three_tensors_axis_1():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int64).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=1)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP8x23)) 
                x3 = Tensor(Dtype.FP8x23, x3.shape,to_fp(
                x3.flatten(), FixedImpl.FP8x23)) 

                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "concat_fp8x23_3d_three_tensors_axis_1"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)
                
            def three_tensors_axis_2():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int64).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=2)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP8x23)) 
                x3 = Tensor(Dtype.FP8x23, x3.shape, to_fp(
                x3.flatten(), FixedImpl.FP8x23)) 

                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "concat_fp8x23_3d_three_tensors_axis_2"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            default()
            axis_1()
            axis_2()
            three_tensors_axis_1()
            three_tensors_axis_2()

        concat_1D()
        concat_2D()
        concat_3D()
    
    staticmethod
    def concat_fp16x16():
        def concat_1D():
            x1 = np.arange(0,3).astype(np.int64)
            x2 = np.arange(3,6).astype(np.int64)
            y = np.concatenate((x1, x2))

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP16x16))  
            x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP16x16)) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

            name = "concat_fp16x16_1d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR.TENSOR)
                    
            
        def concat_2D():
            x1 = np.arange(0,4).astype(np.int64).reshape(2,2)
            x2 = np.arange(4,8).astype(np.int64).reshape(2,2)
            y = np.concatenate((x1, x2), axis=0)

            x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP16x16))
            x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP16x16)) 
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

            name = "concat_fp16x16_2d"
            make_node([x1, x2], [y], name)
            make_test(
                inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                file_name= name, trait_type= Trait.TENSOR)
            
        def concat_3D():
            def default():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=0)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP16x16)) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "concat_fp16x16_3d_default"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 0);", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_1():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=1)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP16x16)) 
                y = Tensor(Dtype.FP16x16, y.shape ,to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "concat_fp16x16_3d_axis_1"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def axis_2():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                y = np.concatenate((x1, x2), axis=2)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP16x16)) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "concat_fp16x16_3d_axis_2"
                make_node([x1, x2], [y], name)
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            def three_tensors_axis_1():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int64).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=1)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP16x16)) 
                x3 = Tensor(Dtype.FP16x16, x3.shape, to_fp(
                x3.flatten(), FixedImpl.FP16x16)) 

                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "concat_fp16x16_3d_three_tensors_axis_1"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 1)", 
                    file_name= name, trait_type= Trait.TENSOR)
                
            def three_tensors_axis_2():
                x1 = np.arange(0,27).astype(np.int64).reshape(3,3,3)
                x2 = np.arange(27,54).astype(np.int64).reshape(3,3,3)
                x3 = np.arange(54,81).astype(np.int64).reshape(3,3,3)

                y = np.concatenate((x1, x2, x3), axis=2)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(
                x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(
                x2.flatten(), FixedImpl.FP16x16)) 
                x3 = Tensor(Dtype.FP16x16, x3.shape, to_fp(
                x3.flatten(), FixedImpl.FP16x16)) 

                y = Tensor(Dtype.FP16x16, y.shape,to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "concat_fp16x16_3d_three_tensors_axis_2"
                make_node([x1, x2, x3], [y], name)
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "TensorTrait::concat(array![input_0, input_1, input_2].span(), 2)", 
                    file_name= name, trait_type= Trait.TENSOR)

            default()
            axis_1()
            axis_2()
            three_tensors_axis_1()
            three_tensors_axis_2()

        concat_1D()
        concat_2D()
        concat_3D()
