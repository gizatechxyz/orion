

import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def col2im(data, image_shape, block_shape, dilations=None, pads=None, strides=None):  # type: ignore
    if dilations is None:
        dilations = [1 for s in image_shape]
    if pads is None:
        pads = [0 for s in image_shape] * 2
    if strides is None:
        strides = [1 for s in image_shape]
    bl = np.prod(block_shape)
    C = data.shape[1] // bl
    data = data.reshape(data.shape[:1] + (C,) + (bl,) + data.shape[2:])
    ks = tuple(block_shape)
    res = None
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            out = col2im_naive_implementation(
                data[n, c, ...], image_shape, ks, dilations, pads, strides
            )
            if res is None:
                new_shape = data.shape[:2] + out.shape
                res = np.empty(new_shape, dtype=data.dtype)
            res[n, c, ...] = out
    return (res,)  # type: ignore

def _get_indices(i, shape):
    res = np.empty((len(shape),), dtype=np.int64)
    k = len(shape) - 1
    while k > 0:
        m = i % shape[k]
        res[k] = m
        i -= m
        i /= shape[k]
        k -= 1
    res[0] = i
    return res

def _col2im_shape_check(X, output_shape, kernel_shape, dilations, pads, strides):  # type: ignore
    n_input_plane = X.shape[0]

    kernel_size = np.prod(kernel_shape)

    if n_input_plane % kernel_size != 0:
        raise ValueError(
            f"Expected size of input's dimension 1 to be divisible by the "
            f"product of kernel_size={kernel_size}, "
            f"but got input.size(1)={n_input_plane} "
            f"and kernel_shape={kernel_shape}, X.shape={X.shape}, output_shape={output_shape}."
        )

    input_length = X.shape[1]
    n_dims = len(output_shape)
    n_blocks = []

    
    for i in range(n_dims):
        n_block = (
            output_shape[i]
            + pads[i, :].sum()
            - dilations[i] * (kernel_shape[i] - 1)
            - 1
        ) // strides[i] + 1
        n_blocks.append(n_block)

    
    block_size = np.prod(n_blocks)
    if input_length != block_size:
        raise ValueError(
            f"Given n_input_plane={n_input_plane}, X.shape={X.shape}, "
            f"output_shape={output_shape}, kernel_shape={kernel_shape}, "
            f"dilations={dilations}, pads={pads}, strides={strides}, "
            f"expected size of input's dimension 2 to match the calculated number of "
            f"sliding blocks {n_blocks} = {block_size}, "
            f"but got input.size(2)={input_length}.",
        )


def col2im_naive_implementation(data, image_shape, kernel_shape, dilations, pads, strides):  # type: ignore

    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    _col2im_shape_check(data, image_shape, kernel_shape, dilations, new_pads, strides)

    data_col = data
    data_im = np.zeros(image_shape, dtype=data.dtype)

    dim_col = []
    for i in range(n_dims):
        col = (
            image_shape[i]
            + new_pads[i, :].sum()
            - (dilations[i] * (kernel_shape[i] - 1) + 1)
        ) // strides[i] + 1
        dim_col.append(col) 
    kernel_size = np.prod(kernel_shape)
    col_size = np.prod(dim_col)
    for c_col in range(kernel_size):
        offset = _get_indices(c_col, kernel_shape)     

        for col in range(col_size):
            
            ind_col = _get_indices(col, dim_col)
            ind_im = []
            for i in range(n_dims):
                ind = (
                    ind_col[i] * strides[i] - new_pads[i, 0] + offset[i] * dilations[i]
                )
                ind_im.append(ind)
            if not _is_out(ind_im, data_im.shape):
                data_im[tuple(ind_im)] += data_col[c_col, col]


    return data_im


def _is_out(ind, shape):  
    for i, s in zip(ind, shape):
        if i < 0:
            return True
        if i >= s:
            return True
    return False
    


class Col2im(RunAll):

    @staticmethod
    def export_col2im() -> None:
        x = np.array(
            [
                [
                    [1.0, 6.0, 11.0, 16.0, 21.0],  # (1, 5, 5)
                    [2.0, 7.0, 12.0, 17.0, 22.0],
                    [3.0, 8.0, 13.0, 18.0, 23.0],
                    [4.0, 9.0, 14.0, 19.0, 24.0],
                    [5.0, 0.0, 15.0, 20.0, 25.0],
                ]
            ]
        ).astype(np.float32)

        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)
    
        y = col2im(x,image_shape,block_shape)
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "col2im"
        func_sig = "NNTrait::col2im("
        func_sig += "@input_0,"
        func_sig += "array![5, 5].span(),"
        func_sig += "array![1, 5].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        

    @staticmethod
    def export_col2im_strides() -> None:
        x = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],  # (1, 9, 4)
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([3, 3]).astype(np.int64)

        y = col2im(x,image_shape,block_shape,strides=[2, 2])
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "col2im_strides"
        func_sig = "NNTrait::col2im("
        func_sig += "@input_0,"
        func_sig += "array![5, 5].span(),"
        func_sig += "array![3, 3].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()))"
        make_test(
            [x], y, func_sig, name, Trait.NN)

    @staticmethod
    def export_col2im_pads() -> None:
        x = np.array(
            [
                [
                    [
                        1.0, 6.0, 11.0, 16.0, 21.0, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71,
                    ],  # (1, 5, 15)
                    [
                        2.0, 7.0, 12.0, 17.0, 22.0, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72,
                    ],
                    [
                        3.0, 8.0, 13.0, 18.0, 23.0, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73,
                    ],
                    [
                        4.0, 9.0, 14.0, 19.0, 24.0, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74,
                    ],
                    [
                        5.0, 10.0, 15.0, 20.0, 25.0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                    ],
                ]
            ]
        ).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)

        y = col2im(x,image_shape,block_shape,pads=[0, 1, 0, 1])
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "col2im_pads"
        func_sig = "NNTrait::col2im("
        func_sig += "@input_0,"
        func_sig += "array![5, 5].span(),"
        func_sig += "array![1, 5].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![0, 1, 0, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_col2im_dilations() -> None:
        x = np.array(
            [
                [
                    [1.0, 5.0, 9.0, 13.0, 17],  # (1, 4, 5)
                    [2.0, 6.0, 10.0, 14.0, 18],
                    [3.0, 7.0, 11.0, 15.0, 19],
                    [4.0, 8.0, 12.0, 16.0, 20],
                ]
            ]
        ).astype(np.float32)
        image_shape = np.array([6, 6]).astype(np.int64)
        block_shape = np.array([2, 2]).astype(np.int64)


        y = col2im(x,image_shape,block_shape, dilations=[1, 5])
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "col2im_dilations"
        func_sig = "NNTrait::col2im("
        func_sig += "@input_0,"
        func_sig += "array![6, 6].span(),"
        func_sig += "array![2, 2].span(),"
        func_sig += "Option::Some(array![1, 5].span()),"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_col2im_5D() -> None:
        x = np.array(
            [
                [
                    [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56],  # (1, 10, 12)
                    [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57],
                    [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58],
                    [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59],
                    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
                    [61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116],
                    [62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117],
                    [63, 68, 73, 78, 83, 88, 93, 98, 103, 108, 113, 118],
                    [64, 69, 74, 79, 84, 89, 94, 99, 104, 109, 114, 119],
                    [65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
                ]
            ]
        ).astype(np.float32)
        image_shape = np.array([3, 4, 5]).astype(np.int64)
        block_shape = np.array([1, 1, 5]).astype(np.int64)
    
        y = col2im(x,image_shape,block_shape)
        y = np.array(y[0])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "col2im_5D"
        func_sig = "NNTrait::col2im("
        func_sig += "@input_0,"
        func_sig += "array![3, 4, 5].span(),"
        func_sig += "array![1, 1, 5].span(),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x], y, func_sig, name, Trait.NN)


        