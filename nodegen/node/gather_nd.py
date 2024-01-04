import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl

def gather_nd_impl(
    data: np.ndarray, indices: np.ndarray, batch_dims: int
) -> np.ndarray:
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # The list of data/indice shape of batch_dims
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array
    batch_dims_size = 1

    # Check the shape of indice and data are identicial for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below

    # Compute shape of output array
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])

    # gather each scalar value from 'data'
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim, *gather_index)])
    return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)

        
class Gather_nd(RunAll):

    @staticmethod
    def gather_nd_fp16x16():
        def gather_nd_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,3)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=0)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "gather_nd_fp16x16_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(0))", 
                    name= name)
                
            def batch_dims1():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=1)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "gather_nd_fp16x16_3d_batch_dims1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(1))", 
                    name= name)
                
            def batch_dims2():
                x1 = np.arange(0,54).reshape(3,3,3,2).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=2)

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "gather_nd_fp16x16_3d_batch_dims2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(2))", 
                    name= name)
                
            default()
            batch_dims1()
            batch_dims2()
        gather_nd_3D()


    @staticmethod
    def gather_nd_fp8x23():
        def gather_nd_3D():
            def default():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,3)).astype(np.int64)
                y = gather_nd_impl(x1, x2, batch_dims=0)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "gather_nd_fp8x23_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(0))", 
                    name= name)
                
            def batch_dims1():
                x1 = np.arange(0,27).reshape(3,3,3).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.int64)
                y = gather_nd_impl(x1, x2, batch_dims=1)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "gather_nd_fp8x23_3d_batch_dims1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(1))", 
                    name= name)
                
            def batch_dims2():
                x1 = np.arange(0,54).reshape(3,3,3,2).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=2)

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "gather_nd_fp8x23_3d_batch_dims2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(2))", 
                    name= name)
                
            default()
            batch_dims1()
            batch_dims2()
        gather_nd_3D()


    @staticmethod
    def gather_nd_i8():
        def gather_nd_3D():
            def default():
                x1 = np.arange(0,9).reshape(3,3).astype(np.int8)
                x2 = np.random.randint(low = 0,high=2, size=(3,2)).astype(np.int8)
                y = gather_nd_impl(x1, x2, batch_dims=0)

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "gather_nd_i8_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(0))", 
                    name= name)
                
            def batch_dims1():
                x1 = np.arange(0,9).reshape(3,3).astype(np.int8)
                x2 = np.random.randint(low = 0,high=2, size=(3,1)).astype(np.int8)
                y = gather_nd_impl(x1, x2, batch_dims=1)

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "gather_nd_i8_3d_batch_dims1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(1))", 
                    name= name)
                
           
            default()
            batch_dims1()
        gather_nd_3D()


    @staticmethod
    def gather_nd_i32():
        def gather_nd_3D():
            def default():
                x1 = np.arange(0,24).reshape(4,2,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=2, size=(3,2)).astype(np.int32)
                y = gather_nd_impl(x1, x2, batch_dims=0)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "gather_nd_i32_3d_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(0))", 
                    name= name)
                
            def batch_dims1():
                x1 = np.arange(0,108).reshape(4,3,3,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=3, size=(4,2,3)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=1)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "gather_nd_i32_3d_batch_dims1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(1))", 
                    name= name)
                
            def batch_dims2():
                x1 = np.arange(0,54).reshape(3,3,3,2).astype(np.int64)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=2)

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "gather_nd_i32_3d_batch_dims2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(2))", 
                    name= name)
                
            default()
            batch_dims1()
            batch_dims2()
        gather_nd_3D()

    @staticmethod
    def gather_nd_u32():
        def gather_nd_3D():
            def default():
                x1 = np.arange(0,108).reshape(3,3,4,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=0)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "gather_nd_u32_default"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(0))", 
                    name= name)
                
            def batch_dims1():
                x1 = np.arange(0,108).reshape(3,3,4,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=1)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "gather_nd_u32_batch_dims1"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(1))", 
                    name= name)
                
            def batch_dims2():
                x1 = np.arange(0,108).reshape(3,3,4,3).astype(np.int32)
                x2 = np.random.randint(low = 0,high=2, size=(3,3,2)).astype(np.uint32)
                y = gather_nd_impl(x1, x2, batch_dims=2)

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "gather_nd_u32_batch_dims2"
                make_test(
                    inputs = [x1, x2], output = y, func_sig = "input_0.gather_nd(indices:input_1, batch_dims:Option::Some(2))", 
                    name= name)
                
            
            default()
            batch_dims1()
            batch_dims2()
        gather_nd_3D()