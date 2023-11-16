import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

# The below ScatterElements' numpy implementation is from https://stackoverflow.com/a/46204790/11767360
def scatter_elements(data, indices, updates, axis=0, reduction="none"):  # type: ignore
    if axis < 0:
        axis = data.ndim + axis

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1 :]

    def make_slice(arr, axis, i):  # type: ignore
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):  # type: ignore
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    def make_indices_for_duplicate(idx):  # type: ignore
        final_idx = []
        for i in range(len(idx[0])):
            final_idx.append(tuple(idx_element[i] for idx_element in idx))
        return list(final_idx)

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced indices for scattering of updates param. in data
    idx = [
        [
            unpack(np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0],
        ]
        for i in range(indices.shape[axis])
    ]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing of elements in the updates
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(
        axis, np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape))
    )

    scattered = np.copy(data)
    if reduction == "none":
        scattered[tuple(idx)] = updates[tuple(updates_idx)]
    else:
        idx, updates_idx = make_indices_for_duplicate(idx), make_indices_for_duplicate(
            updates_idx
        )
        for iter, idx_set in enumerate(idx):
            if reduction == "add":
                scattered[idx_set] += updates[updates_idx[iter]]
            elif reduction == "mul":
                scattered[idx_set] *= updates[updates_idx[iter]]
            elif reduction == "max":
                scattered[idx_set] = np.maximum(
                    scattered[idx_set], updates[updates_idx[iter]]
                )
            elif reduction == "min":
                scattered[idx_set] = np.minimum(
                    scattered[idx_set], updates[updates_idx[iter]]
                )
    return scattered

class Scatter(RunAll):

    @staticmethod
    def scatter_fp16x16():
            
        def scatter():
            def default():
                x1 = np.zeros((3, 3)).astype(np.int64)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int64)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 0, 'none')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(x2.flatten(), FixedImpl.FP16x16))
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_fp16x16_3d_default"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(0), reduction:Option::Some('none'))", 
                    name= name)
                
            def axis_1():
                x1 = np.zeros((3, 3)).astype(np.int64)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int64)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'none')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(x2.flatten(), FixedImpl.FP16x16))
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_fp16x16_3d_axis1"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('none'))", 
                    name= name)
                
            def axis_1_add():
                x1 = np.zeros((3, 3)).astype(np.int64)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int64)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'add')

                x1 = Tensor(Dtype.FP16x16, x1.shape, to_fp(x1.flatten(), FixedImpl.FP16x16))
                x2 = Tensor(Dtype.FP16x16, x2.shape, to_fp(x2.flatten(), FixedImpl.FP16x16))
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16)) 

                name = "scatter_fp16x16_3d_axis1_add"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('add'))", 
                    name= name)
                
            default()
            axis_1()
            axis_1_add()
        scatter()

    @staticmethod
    def scatter_fp8x23():
            
        def scatter():
            def default():

                x1 = np.zeros((3, 3)).astype(np.int64)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int64)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 0, 'none')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(x2.flatten(), FixedImpl.FP8x23))
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "scatter_fp8x23_default"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(0), reduction:Option::Some('none'))", 
                    name= name)
                

            def axis1():
                x1 = np.zeros((3, 3)).astype(np.int64)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int64)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'none')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(x2.flatten(), FixedImpl.FP8x23))
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "scatter_fp8x23_axis1"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('none'))", 
                    name= name)
                
            def axis1_mul():
                x1 = np.zeros((3, 3)).astype(np.int64)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int64)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 0, 'mul')

                x1 = Tensor(Dtype.FP8x23, x1.shape, to_fp(x1.flatten(), FixedImpl.FP8x23))
                x2 = Tensor(Dtype.FP8x23, x2.shape, to_fp(x2.flatten(), FixedImpl.FP8x23))
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))

                name = "scatter_fp8x23_mul"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(0), reduction:Option::Some('mul'))", 
                    name= name)
                
            default()
            axis1()
            axis1_mul()
        scatter()

    @staticmethod
    def scatter_i8():
            
        def scatter_3D():
            def default():
                x1 = np.zeros((3, 3)).astype(np.int8)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int8)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 0, 'none')

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.I8, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "scatter_i8_default"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(0), reduction:Option::Some('none'))", 
                    name= name)
                
            def axis1():
                x1 = np.zeros((3, 3)).astype(np.int8)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int8)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'none')

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.I8, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "scatter_i8_axis1"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('none'))", 
                    name= name)
                
                
            def axis1_max():
                x1 = np.zeros((3, 3)).astype(np.int8)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int8)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'max')

                x1 =  Tensor(Dtype.I8, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.I8, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.I8, y.shape, y.flatten()) 

                name = "scatter_i8_axis1_max"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('max'))", 
                    name= name)
                
            default()
            axis1()
            axis1_max()
        scatter_3D()


    @staticmethod
    def scatter_i32():
        def scatter_3D():
            def default():
                x1 = np.zeros((3, 3)).astype(np.int32)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int32)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 0, 'none')

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.I32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "scatter_i8_default"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(0), reduction:Option::Some('none'))", 
                    name= name)
                
            def axis1():
                x1 = np.zeros((3, 3)).astype(np.int32)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int32)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'none')

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.I32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "scatter_i8_axis1"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('none'))", 
                    name= name)
                
            def axis_min():
                x1 = np.zeros((3, 3)).astype(np.int32)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.int32)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'min')

                x1 =  Tensor(Dtype.I32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.I32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.I32, y.shape, y.flatten()) 

                name = "scatter_i8_default"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('min'))", 
                    name= name)
                
            default()
            axis1()
            axis_min()
        scatter_3D()


    @staticmethod
    def scatter_u32():
            
        def scatter_3D():
            def default():
                x1 = np.zeros((3, 3)).astype(np.uint32)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.uint32)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 0, 'none')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_u32_default"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(0), reduction:Option::Some('none'))", 
                    name= name)
                
                
            def axis1():
                x1 = np.zeros((3, 3)).astype(np.uint32)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.uint32)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 1, 'none')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_u32_axis1"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(1), reduction:Option::Some('none'))", 
                    name= name)
                
            def axis_add():
                x1 = np.zeros((3, 3)).astype(np.uint32)
                x2 = np.arange(1, 10).reshape((3, 3)).astype(np.uint32)
                x3 = np.array(
                        [[0,1,2],
                        [2,0,1],
                        [1,0,1]],
                        )
                y = scatter_elements(x1, x3, x2, 0, 'add')

                x1 =  Tensor(Dtype.U32, x1.shape, x1.flatten())
                x2 =  Tensor(Dtype.U32, x2.shape, x2.flatten())
                x3 = Tensor(Dtype.U32, x3.shape, x3.flatten()) 
                y =  Tensor(Dtype.U32, y.shape, y.flatten()) 

                name = "scatter_u32_add"
                make_test(
                    inputs = [x1, x2, x3], output = y, func_sig = "input_0.scatter(updates:input_1, indices:input_2, axis:Option::Some(0), reduction:Option::Some('add'))", 
                    name= name)
                
            default()
            axis1()
            axis_add()
        scatter_3D()
