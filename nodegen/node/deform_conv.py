import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

import numpy as np

def deform_conv_implementation(  # type: ignore
    X, 
    W,
    offset,
    B=None,
    mask=None,
    dilations=None,
    group=None,
    kernel_shape=None,
    offset_group=None,
    pads=None,
    strides=None,
):
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]
    if group is None:
        group = 1
    if offset_group is None:
        offset_group = 1

    n, ic = X.shape[:2]
    oc = W.shape[0]
    output_shape = offset.shape[2:]

    if ic != W.shape[1] * group or oc % group != 0:
        raise ValueError(
            f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}."
        )
    ics_per_group, ocs_per_group = W.shape[1], oc // group

    if ic % offset_group != 0:
        raise ValueError("Number of input channels must be divisible by offset_group.")
    ics_per_offset_group = ic // offset_group

    if offset_group * np.prod(kernel_shape) * len(kernel_shape) != offset.shape[1]:
        raise ValueError(
            f"Offset shape {offset.shape} is inconsistent with offset_group {offset_group} "
            f"and kernel shape {kernel_shape}."
        )
    offset = offset.reshape(
        (n, offset_group, *kernel_shape, len(kernel_shape), *output_shape)
    )

    if mask is None:
        mask = np.ones((n, offset_group * np.prod(kernel_shape), *output_shape))
    mask = mask.reshape((n, offset_group, *kernel_shape, *output_shape))

    from onnx.reference.ops._op_list import GridSample

    if len(X.shape) == 4:
        ih, iw = X.shape[2:]
        oh, ow = offset.shape[-2:]
        kh, kw = kernel_shape
        sth, stw = strides
        dh, dw = dilations
        kh_new, kw_new = (kh - 1) * dh + 1, (kw - 1) * dw + 1

        if oh != int(((ih - kh_new + pads[0] + pads[2]) / sth) + 1) or ow != int(
            ((iw - kw_new + pads[1] + pads[3]) / stw) + 1
        ):
            raise RuntimeError(
                "Padding, dilation, stride, and kernel shape incompatible with output shape."
            )

        bh, bw = -pads[0], -pads[1]

        res = np.zeros((n, oc, oh, ow), dtype=X.dtype)
        if B is not None:
            res[:, :, :, :] = B.reshape((1, -1, 1, 1))

        kernel_pos_w, kernel_pos_h = np.meshgrid(
            np.arange(0, kw_new, dw), np.arange(0, kh_new, dh)
        )
        
        kernel_pos_wrt_first_elem = np.stack(
            (kernel_pos_h, kernel_pos_w), axis=2
        ) 

        for batch_idx in range(n):
            for oc_idx in range(oc):
                for ic_idx in range(ic):
                    # Group convolution logic
                    if ic_idx // ics_per_group != oc_idx // ocs_per_group:
                        # Input channel and output channel don't belong to same group
                        continue

                    # Offset group logic
                    offset_group_idx = ic_idx // ics_per_offset_group

                    for i in range(oh):
                        h_coord = bh + sth * i
                        for j in range(ow):
                            w_coord = bw + stw * j
                            
                            kernel = np.copy(kernel_pos_wrt_first_elem).astype(float)
                            kernel[:, :, 0] += (
                                h_coord
                                + offset[batch_idx, offset_group_idx, :, :, 0, i, j]
                            )
                            kernel[:, :, 1] += (
                                w_coord
                                + offset[batch_idx, offset_group_idx, :, :, 1, i, j]
                            )

                            kernel[:, :, 0] = kernel[:, :, 0] / (ih - 1) * 2 - 1
                            kernel[:, :, 1] = kernel[:, :, 1] / (iw - 1) * 2 - 1

                            kernel = np.expand_dims(kernel, 0)  
                            
                            kernel = np.flip(
                                kernel, 3
                            )  
                            
                            grid_sample_output = GridSample.eval(
                                X[batch_idx : batch_idx + 1, ic_idx : ic_idx + 1],
                                kernel,
                                align_corners=1,
                            )

                            conv_value = np.multiply(
                                grid_sample_output,
                                W[oc_idx, ic_idx % ics_per_group, :, :],
                            )
                            conv_value = np.multiply(
                                conv_value,
                                mask[batch_idx, offset_group_idx, :, :, i, j],
                            )
                            res[batch_idx, oc_idx, i, j] += np.sum(conv_value)

        return res
    raise RuntimeError(
        f"The convolution for X.shape={X.shape}, W.shape={W.shape}, "
        f"kernel_shape={kernel_shape} is not implemented yet."
    )



def deform_conv_implementation(  # type: ignore
    X, 
    W,
    offset,
    B=None,
    mask=None,
    dilations=None,
    group=None,
    kernel_shape=None,
    offset_group=None,
    pads=None,
    strides=None,
):
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]
    if group is None:
        group = 1
    if offset_group is None:
        offset_group = 1

    n, ic = X.shape[:2]
    oc = W.shape[0]
    output_shape = offset.shape[2:]

    if ic != W.shape[1] * group or oc % group != 0:
        raise ValueError(
            f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}."
        )
    ics_per_group, ocs_per_group = W.shape[1], oc // group

    if ic % offset_group != 0:
        raise ValueError("Number of input channels must be divisible by offset_group.")
    ics_per_offset_group = ic // offset_group

    if offset_group * np.prod(kernel_shape) * len(kernel_shape) != offset.shape[1]:
        raise ValueError(
            f"Offset shape {offset.shape} is inconsistent with offset_group {offset_group} "
            f"and kernel shape {kernel_shape}."
        )
    offset = offset.reshape(
        (n, offset_group, *kernel_shape, len(kernel_shape), *output_shape)
    )

    if mask is None:
        mask = np.ones((n, offset_group * np.prod(kernel_shape), *output_shape))
    mask = mask.reshape((n, offset_group, *kernel_shape, *output_shape))

    from onnx.reference.ops._op_list import GridSample

    if len(X.shape) == 4:
        ih, iw = X.shape[2:]
        oh, ow = offset.shape[-2:]
        kh, kw = kernel_shape
        sth, stw = strides
        dh, dw = dilations
        kh_new, kw_new = (kh - 1) * dh + 1, (kw - 1) * dw + 1

        if oh != int(((ih - kh_new + pads[0] + pads[2]) / sth) + 1) or ow != int(
            ((iw - kw_new + pads[1] + pads[3]) / stw) + 1
        ):
            raise RuntimeError(
                "Padding, dilation, stride, and kernel shape incompatible with output shape."
            )

        bh, bw = -pads[0], -pads[1]

        res = np.zeros((n, oc, oh, ow), dtype=X.dtype)
        if B is not None:
            res[:, :, :, :] = B.reshape((1, -1, 1, 1))

        kernel_pos_w, kernel_pos_h = np.meshgrid(
            np.arange(0, kw_new, dw), np.arange(0, kh_new, dh)
        )
        
        kernel_pos_wrt_first_elem = np.stack(
            (kernel_pos_h, kernel_pos_w), axis=2
        ) 

        for batch_idx in range(n):
            for oc_idx in range(oc):
                for ic_idx in range(ic):
                    # Group convolution logic
                    if ic_idx // ics_per_group != oc_idx // ocs_per_group:
                        # Input channel and output channel don't belong to same group
                        continue

                    # Offset group logic
                    offset_group_idx = ic_idx // ics_per_offset_group

                    for i in range(oh):
                        h_coord = bh + sth * i
                        for j in range(ow):
                            w_coord = bw + stw * j
                            
                            kernel = np.copy(kernel_pos_wrt_first_elem).astype(float)
                            kernel[:, :, 0] += (
                                h_coord
                                + offset[batch_idx, offset_group_idx, :, :, 0, i, j]
                            )
                            kernel[:, :, 1] += (
                                w_coord
                                + offset[batch_idx, offset_group_idx, :, :, 1, i, j]
                            )

                            kernel[:, :, 0] = kernel[:, :, 0] / (ih - 1) * 2 - 1
                            kernel[:, :, 1] = kernel[:, :, 1] / (iw - 1) * 2 - 1

                            kernel = np.expand_dims(kernel, 0)  
                            
                            kernel = np.flip(
                                kernel, 3
                            )  
                            
                            grid_sample_output = GridSample.eval(
                                X[batch_idx : batch_idx + 1, ic_idx : ic_idx + 1],
                                kernel,
                                align_corners=1,
                            )

                            conv_value = np.multiply(
                                grid_sample_output,
                                W[oc_idx, ic_idx % ics_per_group, :, :],
                            )
                            conv_value = np.multiply(
                                conv_value,
                                mask[batch_idx, offset_group_idx, :, :, i, j],
                            )
                            res[batch_idx, oc_idx, i, j] += np.sum(conv_value)

        return res
    raise RuntimeError(
        f"The convolution for X.shape={X.shape}, W.shape={W.shape}, "
        f"kernel_shape={kernel_shape} is not implemented yet."
    )


class Deform_conv(RunAll):
    
    @staticmethod
    def export_deform_conv_without_padding() -> None:
        x = np.arange(9).astype(np.float32)
        x.shape = (1, 1, 3, 3)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)

        # Convolution without padding
        offset = np.zeros((1, 8, 2, 2), dtype=np.float32)
        offset[
            0, 0, 0, 0
        ] = 0.5  
        offset[
            0, 5, 0, 1
        ] = -0.1  


        
        y = deform_conv_implementation(x, w, offset, kernel_shape=[2, 2])
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        offset = Tensor(Dtype.FP16x16, offset.shape, to_fp(offset.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   

        name = "deform_conv"
        func_sig = "NNTrait::deform_conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "@input_2,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w, offset], y, func_sig, name, Trait.NN)
    
    @staticmethod
    def export_deform_conv_with_padding() -> None:
        x = np.arange(9).astype(np.float32)
        x.shape = (1, 1, 3, 3)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)

        # Convolution with padding
        offset = np.zeros((1, 8, 4, 4), dtype=np.float32)
        offset[
            0, 0, 0, 0
        ] = 0.5  
        offset[
            0, 5, 1, 2
        ] = -0.1 


        
        y = deform_conv_implementation(x, w, offset, kernel_shape=[2, 2], pads=[1, 1, 1, 1])
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        offset = Tensor(Dtype.FP16x16, offset.shape, to_fp(offset.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   

        name = "deform_conv_with_padding"
        func_sig = "NNTrait::deform_conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "@input_2,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1, 1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x, w, offset], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_deform_conv_with_mask_bias() -> None:
        x = np.arange(9).astype(np.float32)
        x.shape = (1, 1, 3, 3)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)
        
        b = np.ones((1,), dtype=np.float32)

        offset = np.zeros((1, 8, 2, 2), dtype=np.float32)
        offset[
            0, 0, 0, 0
        ] = 0.5  
        offset[
            0, 5, 0, 1
        ] = -0.1  

        mask = np.ones((1, 4, 2, 2), dtype=np.float32)
        mask[0, 2, 1, 1] = 0.2 
        
        y = deform_conv_implementation(x, w, offset, mask=mask, B=b, kernel_shape=[2, 2])
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        offset = Tensor(Dtype.FP16x16, offset.shape, to_fp(offset.flatten(), FixedImpl.FP16x16))
        b = Tensor(Dtype.FP16x16, b.shape, to_fp(b.flatten(), FixedImpl.FP16x16))
        mask = Tensor(Dtype.FP16x16, mask.shape, to_fp(mask.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   

        name = "deform_conv_with_mask_bias"
        func_sig = "NNTrait::deform_conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "@input_2,"
        func_sig += "Option::Some(input_3.data),"
        func_sig += "Option::Some(input_4),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w, offset, b, mask], y, func_sig, name, Trait.NN)
        
        
    @staticmethod
    def export_deform_conv_with_multiple_offset_groups() -> None:
        x = np.zeros((1, 2, 3, 3), dtype=np.float32)
        x[0, 0] = np.reshape(np.arange(9).astype(np.float32), (3, 3))
        x[0, 1] = np.reshape(np.arange(8, -1, -1).astype(np.float32), (3, 3))
        x.shape = (1, 2, 3, 3)
        w = np.ones((1, 2, 2, 2), dtype=np.float32)

        offset = np.zeros((1, 16, 2, 2), dtype=np.float32)
        offset[
            0, 0, 0, 0
        ] = 0.5  
        offset[
            0, 13, 0, 1
        ] = (
            -0.1
        )  


        y = deform_conv_implementation(x, w, offset, offset_group=2, kernel_shape=[2, 2])
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        offset = Tensor(Dtype.FP16x16, offset.shape, to_fp(offset.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   

        name = "deform_conv_with_multiple_offset_groups"
        func_sig = "NNTrait::deform_conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "@input_2,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()),"
        func_sig += "Option::Some(2),"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w, offset], y, func_sig, name, Trait.NN)



    