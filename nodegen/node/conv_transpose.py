

import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def conv_transpose( 
    X,
    W,
    B=None,
    auto_pad=None,
    dilations=None,
    group=None,
    kernel_shape=None,
    output_padding=None,
    output_shape=None,
    pads=None,
    strides=None,
):
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if output_padding is None:
        output_padding = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]
    if pads is None and auto_pad not in {"SAME_UPPER", "SAME_LOWER"}:
        pads = [0 for i in range(2 * len(strides))]
    if pads is None:
        if output_shape is None:
            output_shape = [
                X.shape[i + 2] * strides[i] for i in range(len(strides))
            ]
        total_padding = [
            strides[i] * (X.shape[i + 2] - 1)
            + output_padding[i]
            + ((kernel_shape[i] - 1) * dilations[i] + 1)
            - output_shape[i]
            for i in range(len(output_shape))
        ]
        pads_1 = []
        pads_2 = []
        for i in range(len(output_shape)):
            if auto_pad == "SAME_UPPER":
                pads_1.append(total_padding[i] // 2)
                pads_2.append(total_padding[i] - (total_padding[i] // 2))
            else:
                pads_1.append(total_padding[i] - (total_padding[i] // 2))
                pads_2.append(total_padding[i] // 2)
        pads = pads_1 + pads_2
        n_dims = len(pads) // 2
    else:
        n_dims = len(X.shape) - 2
        new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
        if output_shape is None:
            output_shape = [
                strides[i] * (X.shape[i + 2] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - new_pads[i, :].sum()
                for i in range(n_dims)
            ]
    kernel_shape = W.shape[2:]
    kernel_size = np.prod(kernel_shape)
    num_output_channels = W.shape[1] * group
    kernel_dim = num_output_channels // group * kernel_size
    C = X.shape[1]  # num_inputs_channels
    m = kernel_dim  # kernel_dim
    n = np.prod(X.shape[2:])  # input_image_size
    k = C // group
    w_reshaped = W.reshape((group, k, m))
    final = None

    # N x C x H x W = X.shape
    # C x M/group x k1 x k2 = W.shape
    if group == 1:
        for image_id in range(X.shape[0]):
            w_t = w_reshaped[0].T
            gemm = np.matmul(w_t, X[image_id].reshape((k, n)))
            gemmc = gemm.reshape((num_output_channels, -1, gemm.shape[-1]))
            for c in range(num_output_channels):
                res = col2im_naive_implementation(
                    gemmc[c], output_shape, kernel_shape, dilations, pads, strides
                )
                if final is None:
                    final = np.empty(
                        X.shape[:1] + (num_output_channels,) + res.shape,
                        dtype=X.dtype,
                    )
                if B is not None:
                    res += B[c]
                final[image_id, c, ...] = res[...]
    else:
        raise NotImplementedError(
                f"Implementation for group={group} > 1 is not available yet."
        )


    return (final.astype(X.dtype),) 



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


class Conv_transpose(RunAll):

    @staticmethod
    def export_conv_transpose() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        w = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        y = conv_transpose(x, w, group=1)[0]

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "conv_transpose"
        func_sig = "NNTrait::conv_transpose("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)
        

    @staticmethod
    def export_convtranspose_1d() -> None:
        x = np.array([[[0.0, 1.0, 2.0]]]).astype(np.float32)  # (1, 1, 3)

        w = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).astype(  # (1, 2, 3)
            np.float32
        )

        y = conv_transpose(x, w, group=1)[0]

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "conv_transpose_1d"
        func_sig = "NNTrait::conv_transpose("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)


    @staticmethod
    def export_convtranspose_3d() -> None:
        x = np.array(
            [
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 3, 4, 5)
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                        ],
                        [
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [30.0, 31.0, 32.0, 33.0, 34.0],
                            [35.0, 36.0, 37.0, 38.0, 39.0],
                        ],
                        [
                            [40.0, 41.0, 42.0, 43.0, 44.0],
                            [45.0, 46.0, 47.0, 48.0, 49.0],
                            [50.0, 51.0, 52.0, 53.0, 54.0],
                            [55.0, 56.0, 57.0, 58.0, 59.0],
                        ],
                    ]
                ]
            ]
        ).astype(np.float32)

        w = np.array(
            [
                [
                    [
                        [
                            [1.0, 1.0, 1.0],  # (1, 2, 3, 3, 3)
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                        ],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ],
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ],
                ]
            ]
        ).astype(np.float32)

        y = conv_transpose(x, w, group=1)[0]

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "conv_transpose_3d"
        func_sig = "NNTrait::conv_transpose("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)



    @staticmethod
    def export_convtranspose_attributes() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        w = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        y = conv_transpose(x, w, group=1)[0]

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "conv_transpose_attributes"
        func_sig = "NNTrait::conv_transpose("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)




    @staticmethod
    def export_convtranspose_pads() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        w = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        y = conv_transpose(x, w, group=1,strides=[3, 2],output_shape=[10, 8], kernel_shape=[3, 3], output_padding=[1, 1],)[0]

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))


        name = "conv_transpose_pads"
        func_sig = "NNTrait::conv_transpose("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(array![3, 3].span()),"
        func_sig += "Option::Some(array![1, 1].span()),"
        func_sig += "Option::Some(array![10, 8].span()),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![3, 2].span()))"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)

    @staticmethod
    def export_convtranspose_dilations() -> None:
        x = np.array(
            [[[[3.0, 8.0, 1.0], [9.0, 5.0, 7.0], [3.0, 2.0, 6.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)
        w = np.array([[[[7.0, 2.0], [1.0, 9.0]]]]).astype(np.float32)  # (1, 1, 2, 2)

        y = conv_transpose(x, w, group=1,  dilations=[2, 2])[0]

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "conv_transpose_dilations"
        func_sig = "NNTrait::conv_transpose("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span())," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)


    @staticmethod
    def export_convtranspose_autopad_same() -> None:
        x = np.array(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
        ).astype(np.float32)

        w = np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ]
        ).astype(np.float32)

        y = conv_transpose(x, w, group=1, auto_pad="SAME_UPPER", strides=[2, 2])[0]

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "conv_transpose_autopad_same"
        func_sig = "NNTrait::conv_transpose("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(AUTO_PAD::SAME_UPPER),"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()))"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)
        






