import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl

def scatter_nd_impl(data, indices, updates, reduction="none"):  # type: ignore
    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1] :]

    # Compute output
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        # NOTE: The order of iteration in this loop is not specified.
        if reduction == "add":
            output[tuple(indices[i])] += updates[i]
        elif reduction == "mul":
            output[tuple(indices[i])] *= updates[i]
        elif reduction == "max":
            output[tuple(indices[i])] = np.maximum(output[indices[i]], updates[i])
        elif reduction == "min":
            output[tuple(indices[i])] = np.minimum(output[indices[i]], updates[i])
        else:
            output[tuple(indices[i])] = updates[i]
    return output

data = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            ],
            dtype=np.float32,
        )
indices = np.array([[0], [2]], dtype=np.int64)
updates = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            ],
            dtype=np.float32,
        )

class Scatter_nd(RunAll):

    @staticmethod
    def scatter_nd_fp16x16():
        def scatter_nd_3D():
            def default():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='none')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP16x16, x3.shape, to_fp(x3.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_nd_fp16x16_3d_default"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::None(()))", 
                    name= name)
                
            def add():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='add')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP16x16, x3.shape, to_fp(x3.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_nd_fp16x16_3d_add"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('add'))", 
                    name= name)
                
            def mul():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='mul')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP16x16, x3.shape, to_fp(x3.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_nd_fp16x16_3d_mul"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('mul'))", 
                    name= name)
                
            def max():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='max')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP16x16, x3.shape, to_fp(x3.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_nd_fp16x16_3d_max"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('max'))", 
                    name= name)
                
            def min():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='min')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP16x16, x3.shape, to_fp(x3.flatten(), FixedImpl.FP16x16))
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_nd_fp16x16_3d_min"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('min'))", 
                    name= name)
                
            default()
            add()
            mul()
            max()
            min()
        scatter_nd_3D()


    @staticmethod
    def scatter_nd_fp8x23():
        def scatter_nd_3D():
            def default():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='none')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP8x23, x3.shape, to_fp(x3.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "scatter_nd_fp8x23_3d_default"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::None(()))", 
                    name= name)
                
            def add():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='add')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP8x23, x3.shape, to_fp(x3.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "scatter_nd_fp8x23_3d_add"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('add'))", 
                    name= name)
                
            def mul():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='mul')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP8x23, x3.shape, to_fp(x3.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "scatter_nd_fp8x23_3d_mul"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('mul'))", 
                    name= name)
                
            def max():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='max')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP8x23, x3.shape, to_fp(x3.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "scatter_nd_fp8x23_3d_max"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('max'))", 
                    name= name)
                
            def min():
                x1 = data.astype(np.int64)
                x2 = indices.astype(np.int64)
                x3 = updates.astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='min')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.U32, x2.shape, x2.flatten()) 
                x3 = Tensor(Dtype.FP8x23, x3.shape, to_fp(x3.flatten(), FixedImpl.FP8x23))
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23)) 

                name = "scatter_nd_fp8x23_3d_min"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('min'))", 
                    name= name)
                
            default()
            add()
            mul()
            max()
            min()
        scatter_nd_3D()

    @staticmethod
    def scatter_nd_u32():
        def scatter_nd_3D():
            def default():
                x1 =  np.arange(0,12).reshape((4,3)).astype(np.int32)
                x2 = np.array([[0],[1]]).astype(np.uint32)
                x3 = np.random.randint(low = 0,high=100, size=(2,3)).astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='none')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 =  Tensor(Dtype.U32, x3.shape, x3.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_nd_u32_default"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::None(()))", 
                    name= name)
                
            def add():
                x1 =  np.arange(0,12).reshape((4,3)).astype(np.int32)
                x2 = np.array([[1],[0]]).astype(np.uint32)
                x3 = np.random.randint(low = 0,high=100, size=(2,3)).astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='add')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 =  Tensor(Dtype.U32, x3.shape, x3.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_nd_u32_add"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('add'))", 
                    name= name)
                
            def mul():
                x1 =  np.arange(0,12).reshape((4,3)).astype(np.int32)
                x2 =np.array([[0],[1]]).astype(np.uint32)
                x3 = np.random.randint(low = 0,high=100, size=(2,3)).astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='mul')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 =  Tensor(Dtype.U32, x3.shape, x3.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_nd_u32_mul"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('mul'))", 
                    name= name)
                
            def max():
                x1 =  np.arange(0,12).reshape((4,3)).astype(np.int32)
                x2 =np.array([[0],[1]]).astype(np.uint32)
                x3 = np.random.randint(low = 0,high=100, size=(2,3)).astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='max')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 =  Tensor(Dtype.U32, x3.shape, x3.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_nd_u32_max"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('max'))", 
                    name= name)
                
            def min():
                x1 =  np.arange(0,12).reshape((4,3)).astype(np.int32)
                x2 = np.array([[0],[1]]).astype(np.uint32)
                x3 = np.random.randint(low = 0,high=100, size=(2,3)).astype(np.uint32)
                y = scatter_nd_impl(x1, x2, x3, reduction='min')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 =  Tensor(Dtype.U32, x3.shape, x3.flatten())
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_nd_u32_min"
                make_test(
                    inputs = [x1, x3, x2], output = y, func_sig = "input_0.scatter_nd(updates:input_1, indices:input_2, reduction:Option::Some('min'))", 
                    name= name)
                
            default()
            add()
            mul()
            max()
            min()
        scatter_nd_3D()


   