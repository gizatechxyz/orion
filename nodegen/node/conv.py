import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

import numpy as np

def r_index_check(r_index, shape_out):
    for i in range(len(r_index)):
        if r_index[i] >= shape_out[i]:
            return False
    return True    

def stride(arr):
    stride = np.zeros(len(arr))
    acc = 1
    for i in range(len(arr)):
        stride[i] = acc
        acc *= arr[-(i + 1)]
    return np.flip(stride) 
        
def conv(
        X,
        W,
        B=None,
        auto_pad=None,
        dilations=None,
        group=None,
        kernel_shape=None,
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
        group=1

    if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
        raise ValueError(
            f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}, "
            f"W should be {(W.shape[0], X.shape[1] // group, np.prod(W.shape[1:]) // X.shape[1] * group)}."
        )
    if group > 1:
        res = []
        td = 0
        mg = W.shape[0] // group
        dw = W.shape[1]

        for b in range(X.shape[0]):
            for g in range(group):
                gx = X[b : b + 1, g * dw : (g + 1) * dw]
                gw = W[g * mg : (g + 1) * mg]
                try:
                    cv = conv(
                        gx,
                        gw,
                        None,
                        auto_pad,
                        dilations,
                        1,
                        kernel_shape,
                        pads,
                        strides,
                    )
                except (ValueError, RuntimeError) as e:
                    raise ValueError(
                        f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={g}/{group}, "
                        f"gx.shape={gx.shape}, gw.shape={gw.shape}, auto_pad={auto_pad}, "
                        f"dilations={dilations}, kernel_shape={kernel_shape}, pads={pads}, "
                        f"strides={strides}."
                    ) from e
                if b == 0:
                    td += cv.shape[1]
                res.append((b, cv))

        new_shape = [X.shape[0], *list(res[0][1].shape[1:])]

        new_shape[1] = td
        final = np.zeros(tuple(new_shape), dtype=res[0][1].dtype)
        p = 0
        for b, cv in res:
            final[b : b + 1, p : p + cv.shape[1]] = cv
            p += cv.shape[1]
            if p >= final.shape[1]:
                p = 0
        if B is not None:
            new_shape = [1 for s in final.shape]
            new_shape[1] = B.shape[0]
            b = B.reshape(tuple(new_shape))
            final += b
        return final

    if dilations[0] != 1 or min(dilations) != max(dilations):
        # Let's compute the dilated kernel.
        nd = len(dilations)
        new_kernel_shape = []
        new_shape = list(W.shape[:-nd])
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            new_shape.append(W.shape[di] + (W.shape[di] - 1) * (d - 1))
            new_kernel_shape.append(kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1))
        new_w = np.zeros(tuple(new_shape), dtype=W.dtype)
        indices = [slice(0, new_w.shape[0]), slice(0, new_w.shape[1])]
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            indices.append(slice(0, new_w.shape[di], d))
        new_w[tuple(indices)] = W
        W = new_w
        kernel_shape = new_kernel_shape

    if auto_pad in {"SAME_LOWER", "SAME_UPPER", "VALID"}:
        head = []
        tail = []
        for i in range(len(X.shape) - 2):
            d = X.shape[i]
            target_size = (d + strides[i] - 1) // strides[i]
            pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
            if auto_pad == "SAME_LOWER":
                pad_head = (pad_needed + 1) // 2
            else:
                pad_head = pad_needed // 2
            pad_tail = pad_needed - pad_head
            head.append(pad_head)
            tail.append(pad_tail)
        pads = head + tail

    if len(X.shape) == 3:
        sN, sC, sH = X.shape
        # M, C_group, kH, kW = W.shape
        (kh,) = kernel_shape
        (sth,) = strides

        h_out = int(((sH - kh + pads[0] + pads[1]) / sth) + 1)

        h0 = pads[0]
        oh = -1 * (kh % 2)
        bh = -h0
        eh = h_out * sth
        res = np.zeros((X.shape[0], W.shape[0], h_out))  # type: ignore[assignment]
        if B is not None:
            res[:, :, :] += B.reshape((1, -1, 1))  # type: ignore

        for n in range(0, sN):
            for nw in range(W.shape[0]):
                for c in range(0, sC):
                    w = W[nw : nw + 1, c : c + 1]
                    for io in range(bh, eh, sth):
                        hr = (io - bh) // sth
                        if hr >= h_out:
                            continue
                        i = io + kh % 2
                        ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
                        img = X[n : n + 1, c : c + 1, ih1:ih2]
                        if img.shape != w.shape:
                            jh1, jh2 = max(-oh - i, 0), min(kh, kh + sH - (i + oh + kh))
                            w_ = w[:1, :1, jh1:jh2]
                            
                            if img.shape != w_.shape:
                                raise RuntimeError(
                                    f"Unexpected shape {img.shape} != {w_.shape}, oh={oh}, "
                                    f"i={i}, kh={kh}, sH={sH}, sth={sth}."
                                )
                            s = np.dot(img.reshape((1, -1)), w_.reshape((-1, 1)))[
                                0, 0
                            ]  # (img * w_).sum()
                        else:
                            s = np.dot(img.reshape((1, -1)), w.reshape((-1, 1)))[
                                0, 0
                            ]  # (img * w).sum()
                        res[n, nw, hr] += s  # type: ignore

        return res

    if len(X.shape) == 4:
        sN, sC, sH, sW = X.shape
        # M, C_group, kH, kW = W.shape
        kh, kw = kernel_shape
        sth, stw = strides

        h_out = int(((sH - kh + pads[0] + pads[2]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[3]) / stw) + 1)

        h0, w0 = pads[0], pads[1]
        oh, ow = -1 * (kh % 2), -1 * (kw % 2)
        bh, bw = -h0, -w0
        eh, ew = h_out * sth, w_out * stw
        res = np.zeros((X.shape[0], W.shape[0], h_out, w_out))  # type: ignore[assignment]
        if B is not None:
            res[:, :, :, :] = B.reshape((1, -1, 1, 1))  # type: ignore

        for n in range(0, sN):
            for nw in range(W.shape[0]):
                for c in range(0, sC):
                    w = W[nw : nw + 1, c : c + 1]
                    for io in range(bh, eh, sth):
                        hr = (io - bh) // sth
                        if hr >= h_out:
                            continue
                        i = io + kh % 2
                        ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
                        for jo in range(bw, ew, stw):
                            wr = (jo - bw) // stw
                            if wr >= w_out:
                                continue
                            
                            j = jo + kw % 2
                            iw1, iw2 = max(0, j + ow), min(j + ow + kw, sW)
                            img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2]

                            if img.shape != w.shape:
                                jh1, jh2 = max(-oh - i, 0), min(
                                    kh, kh + sH - (i + oh + kh)
                                )
                                jw1, jw2 = max(-ow - j, 0), min(
                                    kw, kw + sW - (j + ow + kw)
                                )
                                w_ = w[:1, :1, jh1:jh2, jw1:jw2]
                                if img.shape != w_.shape:
                                    raise RuntimeError(
                                        f"Unexpected shape {img.shape} != {w_.shape}, oh={oh}, ow={ow}, "
                                        f"i={i}, j={j}, kh={kh}, kw={kw}, sH={sH}, sW={sW}, sth={sth}, stw={stw}."
                                    )
                                s = np.dot(img.reshape((1, -1)), w_.reshape((-1, 1)))[
                                    0, 0
                                ]  # (img * w_).sum()
                            else:
                                s = np.dot(img.reshape((1, -1)), w.reshape((-1, 1)))[
                                    0, 0
                                ]  # (img * w).sum()
                            res[n, nw, hr, wr] += s  # type: ignore

        return res

    if len(X.shape) == 5:
        sN, sC, sH, sW, sZ = X.shape
        kh, kw, kz = kernel_shape
        sth, stw, stz = strides

        h_out = int(((sH - kh + pads[0] + pads[3]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[4]) / stw) + 1)
        z_out = int(((sZ - kz + pads[2] + pads[5]) / stz) + 1)

        h0, w0, z0 = pads[0], pads[1], pads[2]
        oh, ow, oz = -1 * (kh % 2), -1 * (kw % 2), -1 * (kz % 2)
        bh, bw, bz = -h0, -w0, -z0
        eh, ew, ez = h_out * sth, w_out * stw, z_out * stz
        res = np.zeros((X.shape[0], W.shape[0], h_out, w_out, z_out))  # type: ignore[assignment]
        if B is not None:
            res[:, :, :, :, :] = B.reshape((1, -1, 1, 1, 1))  # type: ignore

        for n in range(0, sN):
            for nw in range(W.shape[0]):
                for c in range(0, sC):
                    w = W[nw : nw + 1, c : c + 1]
                    for io in range(bh, eh, sth):
                        hr = (io - bh) // sth
                        if hr >= h_out:
                            continue
                        i = io + kh % 2
                        ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
                        for jo in range(bw, ew, stw):
                            wr = (jo - bw) // stw
                            if wr >= w_out:
                                continue
                            j = jo + kw % 2
                            iw1, iw2 = max(0, j + ow), min(j + ow + kw, sW)
                            for zo in range(bz, ez, stz):
                                zr = (zo - bz) // stz
                                if zr >= z_out:
                                    continue
                                z = zo + kz % 2
                                iz1, iz2 = max(0, z + oz), min(z + oz + kz, sZ)
                                img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2, iz1:iz2]
                                
                                ### ICI
                                if img.shape != w.shape:
                                    jh1, jh2 = max(-oh - i, 0), min(
                                        kh, kh + sH - (i + oh + kh)
                                    )
                                    jw1, jw2 = max(-ow - j, 0), min(
                                        kw, kw + sW - (j + ow + kw)
                                    )
                                    jz1, jz2 = max(-oz - z, 0), min(
                                        kz, kz + sZ - (z + oz + kz)
                                    )
                                    w_ = w[:1, :1, jh1:jh2, jw1:jw2, jz1:jz2]
                                    if img.shape != w_.shape:
                                        raise RuntimeError(
                                            f"Unexpected shape {img.shape} != {w_.shape}, oh={oh}, ow={ow}, oz={oz}, "
                                            f"i={i}, j={j}, z={z}, kh={kh}, kw={kw}, kz={kz}, "
                                            f"sH={sH}, sW={sW}, sZ={sZ}, sth={sth}, stw={stw}, stz={stz}."
                                        )
                                        
                                    s = np.dot(
                                        img.reshape((1, -1)), w_.reshape((-1, 1))
                                    )[
                                        0, 0
                                    ]  
                                else:
                                    
                                    s = np.dot(
                                        img.reshape((1, -1)), w.reshape((-1, 1))
                                    )[
                                        0, 0
                                    ]  
                                res[n, nw, hr, wr, zr] += s  # type: ignore

        return res

    else:
        nd = len(X.shape[2:])
        sN, sC = X.shape[:2]
        
        x_stride = stride(X.shape)
        w_stride = stride(W.shape)
        x_flatten = X.reshape(int(x_stride[0] * X.shape[0]))
        
        
        shape_out = [int(((X.shape[2+i] - kernel_shape[i] + pads[i] + pads[i + nd]) / strides[i]) + 1) for i in range(nd)]
        o_index = [-1 * (kernel_shape[i] % 2) for i in range(nd)]
        b_index = [-pads[i] for i in range(nd)]
        e_index = [shape_out[i] * strides[i] for i in range(nd)]
        
        
        range_len = [e_index[i] - b_index[i] / strides[i] for i in range(nd)]
        range_stride = stride(range_len)
        
        res_shape = [X.shape[0], W.shape[0]] + shape_out
        res = np.zeros(res_shape) 
        
        res_strides = stride(res_shape)
        if B is not None:
            res[:, :, :, :, :] = B.reshape((1, -1, 1, 1, 1))  # type: ignore

        for n in range(0, sN):
            for nw in range(W.shape[0]):
                for c in range(0, sC):
                    w = W[nw : nw + 1, c : c + 1]
                    for i in range(int(range_len[0] * range_stride[0])):
                        flatten_index = i
                        io_index = np.zeros(nd)
                        r_index = np.zeros(nd)
                        for nx in range(nd):    
                            n_index, rem = divmod(flatten_index, range_stride[nx])
                            flatten_index = rem
                            io_index[nx] = n_index * strides[nx] + b_index[nx]
                            r_index[nx] = n_index
                        if r_index_check(r_index, shape_out):
                            indices = [io_index[nx] + (kernel_shape[nx] % 2) for nx in range(nd)]
                            i1_index = [max(0, indices[nx] + o_index[nx]) for nx in range(nd)]
                            i2_index = [min(X.shape[2 + nx], indices[nx] + o_index[nx] + kernel_shape[nx]) for nx in range(nd)]
                            idiff_index = [int(i2_index[nx] - i1_index[nx]) for nx in range(nd - 1)]
                        
                            i_stride = stride(idiff_index)
                            img = []
                            for ii in range(int(i_stride[0] * idiff_index[0])):
                                flatten_index = ii
                                start = n * x_stride[0] + c * x_stride[1]
                                for nx in range(nd - 1):    
                                    ii_index, rem = divmod(flatten_index, i_stride[nx])
                                    flatten_index = rem
                                    start += (i1_index[nx] + ii_index) * x_stride[2 + nx]
                                start += i1_index[nd-1]   
                                end = start + (i2_index[nd-1] - i1_index[nd-1])
                                img.append(x_flatten[int(start):int(end)])
                            img_shape = [1, 1] + idiff_index
                            w = w.reshape(np.prod(kernel_shape))
                            if len(img) != len(w):
                                j1_index = [max(0, -indices[nx] - o_index[nx]) for nx in range(nd)]
                                j2_index = [min(X.shape[2 + nx] - indices[nx] - o_index[nx], kernel_shape[nx]) for nx in range(nd)]
                                jdiff_index = [j2_index[nx] - j1_index[nx] for nx in range(nd - 1)]


                                w_ = []

                                j_stride = stride(jdiff_index)

                                for jj in range(int(j_stride[0] * jdiff_index[0])):
                                    flatten_index = jj
                                    start = 0
                                    for nx in range(nd):    
                                        jj_index, rem = divmod(flatten_index, range_stride[nx])
                                        flatten_index = rem
                                        start += (j1_index[nx] + jj_index) * kernel_shape[nx]
                                    w_.append(w[int(start + j1_index[-1]):int(start + j1_index[-1] + j2_index[nd-1] - j1_index[nd-1])])
                                
                                
                                img = np.array(img)                                
                                s = np.dot(
                                    np.array(img).reshape((1, -1)), np.array(w_).reshape((-1, 1))
                                )[
                                    0, 0
                                ]  
                            else:
                                img = np.array(img)
                                s = np.dot(
                                    np.array(img).reshape((1, -1)), np.array(w_).reshape((-1, 1))
                                )[
                                    0, 0
                                ]  

                            res_index = []
                            for nx in range(nd):
                                res_index.append(int(r_index[nx]))

                            index = tuple([n, nw]) + tuple(res_index)
                            res[index] += s  # type: ignore
            return res



class Conv(RunAll):
    
    @staticmethod
    def export_conv_1D_no_padding() -> None:
        x = np.array(
            [
                [
                    [
                        0.0, 1.0, 2.0, 3.0, 4.0
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        1.0, 1.0, 1.0
                    ]
                ]
            ]
        ).astype(np.float32)

        
        y = conv(x, w, group = 1)
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   

        name = "conv_1D_no_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
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
    def export_conv_1D_with_padding() -> None:
        x = np.array(
            [
                [
                    [
                        0.0, 1.0, 2.0, 3.0, 4.0
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        1.0, 1.0, 1.0
                    ]
                ]
            ]
        ).astype(np.float32)

        
        y = conv(x, w, group = 1, pads=[1, 1])
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   

        name = "conv_1D_with_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)
        
    @staticmethod
    def export_conv_2D_no_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0],  
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ]
            ]
        ).astype(np.float32)

        
        y = conv(x, w, group = 1, kernel_shape=[3, 3],pads=[0, 0, 0, 0],)
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_2D_with_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
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
    def export_con_2D_with_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0], 
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ]
            ]
        ).astype(np.float32)

        y = conv(x, w, group = 1, kernel_shape=[3, 3],pads=[1, 1, 1, 1],)
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_2D_with_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1, 1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)
        
        
    @staticmethod
    def export_conv_3D_no_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [
                            [  0,   1,   2,   3,   4],[  5,   6,   7,   8,   9],[ 10,  11,  12,  13,  14],[ 15,  16,  17,  18,  19],[ 20,  21,  22,  23,  24]
                        ],
                        [
                            [ 25,  26,  27,  28,  29],[ 30,  31,  32,  33,  34],[ 35,  36,  37,  38,  39],[ 40,  41,  42,  43,  44],[ 45,  46,  47,  48,  49]
                        ],
                        [
                            [ 50,  51,  52,  53,  54],[ 55,  56,  57,  58,  59],[ 60,  61,  62,  63,  64],[ 65,  66,  67,  68,  69],[ 70,  71,  72,  73,  74]
                        ],
                        [
                            [ 75,  76,  77,  78,  79],[ 80,  81,  82,  83,  84],[ 85,  86,  87,  88,  89],[ 90,  91,  92,  93,  94],[ 95,  96,  97,  98,  99]
                        ],
                        [
                            [100, 101, 102, 103, 104],[105, 106, 107, 108, 109],[110, 111, 112, 113, 114],[115, 116, 117, 118, 119],[120, 121, 122, 123, 124]
                        ]
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [
                            [1., 1., 1.],[1., 1., 1.],[1., 1., 1.]
                        ],
                        [
                            [1., 1., 1.],[1., 1., 1.],[1., 1., 1.]
                        ],
                        [
                            [1., 1., 1.],[1., 1., 1.],[1., 1., 1.]
                        ]
                    ]
                ]
            ]
        ).astype(np.float32)
        
        y = conv(x, w, group = 1)
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_3D_no_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
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
    def export_conv_3D_with_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [
                            [  0,   1,   2,   3,   4],[  5,   6,   7,   8,   9],[ 10,  11,  12,  13,  14],[ 15,  16,  17,  18,  19],[ 20,  21,  22,  23,  24]
                        ],
                        [
                            [ 25,  26,  27,  28,  29],[ 30,  31,  32,  33,  34],[ 35,  36,  37,  38,  39],[ 40,  41,  42,  43,  44],[ 45,  46,  47,  48,  49]
                        ],
                        [
                            [ 50,  51,  52,  53,  54],[ 55,  56,  57,  58,  59],[ 60,  61,  62,  63,  64],[ 65,  66,  67,  68,  69],[ 70,  71,  72,  73,  74]
                        ],
                        [
                            [ 75,  76,  77,  78,  79],[ 80,  81,  82,  83,  84],[ 85,  86,  87,  88,  89],[ 90,  91,  92,  93,  94],[ 95,  96,  97,  98,  99]
                        ],
                        [
                            [100, 101, 102, 103, 104],[105, 106, 107, 108, 109],[110, 111, 112, 113, 114],[115, 116, 117, 118, 119],[120, 121, 122, 123, 124]
                        ]
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [
                            [1., 1., 1.],[1., 1., 1.],[1., 1., 1.]
                        ],
                        [
                            [1., 1., 1.],[1., 1., 1.],[1., 1., 1.]
                        ],
                        [
                            [1., 1., 1.],[1., 1., 1.],[1., 1., 1.]
                        ]
                    ]
                ]
            ]
        ).astype(np.float32)
        
        y = conv(x, w, group = 1, pads=[1, 1, 1, 1, 1, 1])
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_3D_with_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1, 1, 1, 1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)  
        
    @staticmethod
    def export_conv_4D_no_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [
                            [
                                [ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8]
                            ],
                            [
                                [ 9, 10, 11],[12, 13, 14],[15, 16, 17]
                            ],
                            [
                                [18, 19, 20],[21, 22, 23],[24, 25, 26]
                            ]
                        ],
                        [
                            [
                                [27, 28, 29],[30, 31, 32],[33, 34, 35]
                            ],
                            [
                                [36, 37, 38],[39, 40, 41],[42, 43, 44]
                            ],
                            [
                                [45, 46, 47],[48, 49, 50],[51, 52, 53]
                            ]
                        ],
                        [
                            [
                                [54, 55, 56],[57, 58, 59],[60, 61, 62]
                            ],
                            [
                                [63, 64, 65],[66, 67, 68],[69, 70, 71]
                            ],
                            [
                                [72, 73, 74],[75, 76, 77],[78, 79, 80]
                            ]
                        ]
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [
                            [
                                [1., 1.],[1., 1.]
                            ],
                            [
                                [1., 1.],[1., 1.]
                            ]
                        ],
                        [
                            [
                                [1., 1.],[1., 1.]
                            ],
                            [
                                [1., 1.],[1., 1.]
                            ]
                        ]
                    ]
                ]
            ]
        ).astype(np.float32)
        
        y = conv(x, w, group = 1)
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_4D_no_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
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
    def export_conv_4D_with_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [
                            [
                                [ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8]
                            ],
                            [
                                [ 9, 10, 11],[12, 13, 14],[15, 16, 17]
                            ],
                            [
                                [18, 19, 20],[21, 22, 23],[24, 25, 26]
                            ]
                        ],
                        [
                            [
                                [27, 28, 29],[30, 31, 32],[33, 34, 35]
                            ],
                            [
                                [36, 37, 38],[39, 40, 41],[42, 43, 44]
                            ],
                            [
                                [45, 46, 47],[48, 49, 50],[51, 52, 53]
                            ]
                        ],
                        [
                            [
                                [54, 55, 56],[57, 58, 59],[60, 61, 62]
                            ],
                            [
                                [63, 64, 65],[66, 67, 68],[69, 70, 71]
                            ],
                            [
                                [72, 73, 74],[75, 76, 77],[78, 79, 80]
                            ]
                        ]
                    ]
                ]
            ]
    ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [
                            [
                                [1., 1.],[1., 1.]
                            ],
                            [
                                [1., 1.],[1., 1.]
                            ]
                        ],
                        [
                            [
                                [1., 1.],[1., 1.]
                            ],
                            [
                                [1., 1.],[1., 1.]
                            ]
                        ]
                    ]
                ]
            ]
        ).astype(np.float32)
        
        y = conv(x, w, group = 1, pads=[1, 1, 1, 1, 1, 1, 1, 1])
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   #
        
        name = "conv_4D_with_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![1, 1, 1, 1, 1, 1, 1, 1].span()),"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN)
        
        
    
    @staticmethod
    def export_conv_with_autopad_same() -> None:
        x = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0],  
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ]
            ]
        ).astype(np.float32)  
        
        y = conv(x, w, group = 1, kernel_shape=[3, 3],auto_pad="SAME_LOWER",strides = [2, 2])
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_2D_with_autopad_same"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::Some(AUTO_PAD::SAME_LOWER)," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![3, 3].span()),"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![2, 2].span()))"
        make_test(
            [x, w], y, func_sig, name, Trait.NN) 
        
    @staticmethod
    def export_conv_with_strides_asymmetric_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0, 29.0],
                        [30.0, 31.0, 32.0, 33.0, 34.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0],  
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ]
            ]
        ).astype(np.float32)

        y = conv(x, w, group = 1, kernel_shape=[3, 3],pads=[1, 0, 1, 0],strides = [2, 2])
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_2D_with_strides_asymmetric_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![3, 3].span()),"
        func_sig += "Option::Some(array![1, 0, 1, 0].span()),"
        func_sig += "Option::Some(array![2, 2].span()))"
        make_test(
            [x, w], y, func_sig, name, Trait.NN) 
        
    @staticmethod
    def export_conv_with_strides_with_padding() -> None:
        x = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0], 
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0, 29.0],
                        [30.0, 31.0, 32.0, 33.0, 34.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        w = np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0], 
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        
        y = conv(x, w, group = 1, kernel_shape=[3, 3],pads=[1, 1, 1, 1],strides = [2, 2])
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))   
        
        name = "conv_2D_with_strides_with_padding"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::Some(array![3, 3].span()),"
        func_sig += "Option::Some(array![1, 1, 1, 1].span()),"
        func_sig += "Option::Some(array![2, 2].span()))"
        make_test(
            [x, w], y, func_sig, name, Trait.NN) 
        
    @staticmethod
    def export_conv_with_2_groups() -> None:
        x = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [
                        [9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]
                ]
            ]
        ).astype(np.float32)
        w =  np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 
                ], 
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 
                ]
            ]
        ).astype(np.float32)
        y = conv(x, w, group = 2)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        w = Tensor(Dtype.FP16x16, w.shape, to_fp(w.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16)) 
        
        name = "conv_2D_with_2_groups"
        func_sig = "NNTrait::conv("
        func_sig += "@input_0,"
        func_sig += "@input_1,"
        func_sig += "Option::None," 
        func_sig += "Option::None," 
        func_sig += "Option::None,"
        func_sig += "Option::Some(2),"
        func_sig += "Option::None,"
        func_sig += "Option::None,"
        func_sig += "Option::None)"
        make_test(
            [x, w], y, func_sig, name, Trait.NN) 

    