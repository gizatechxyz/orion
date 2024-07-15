use core::debug::PrintTrait;

use orion::numbers::NumberTrait;
use orion::numbers::{U32IntoI32, I32IntoU32, I32Div, I32Number};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};
use orion::operators::tensor::core::{stride};

#[derive(Copy, Drop)]
enum AUTO_PAD {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID
}

fn conv<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +AddEq<T>,
    +PrintTrait<T>,
>(
    X: @Tensor<T>,
    W: @Tensor<T>,
    B: Option<Span<T>>,
    auto_pad: Option<AUTO_PAD>,
    dilations: Option<Span<usize>>,
    group: Option<usize>,
    kernel_shape: Option<Span<usize>>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
) -> Tensor<T> {
    let nd = (*X).shape.len() - 2;

    assert((*X).shape.len() >= 3, 'X must have at least 3 dim');

    let dilations = match dilations {
        Option::Some(dilations) => dilations,
        Option::None => {
            let mut dilations: Array<usize> = array![];
            let mut i = 2;
            while i != (*X).shape.len() {
                dilations.append(1);
                i += 1;
            };

            dilations.span()
        },
    };
    let kernel_shape = match kernel_shape {
        Option::Some(kernel_shape) => kernel_shape,
        Option::None => {
            let mut kernel_shape: Array<usize> = array![];
            let mut i = 2;
            while i != (*W).shape.len() {
                kernel_shape.append(*(*W).shape.at(i));
                i += 1;
            };

            kernel_shape.span()
        },
    };
    let pads = match pads {
        Option::Some(pads) => pads,
        Option::None => {
            let mut pads: Array<usize> = array![];
            let mut i = 2;
            while i != (*X).shape.len() {
                pads.append(0);
                pads.append(0);
                i += 1;
            };

            pads.span()
        },
    };
    let strides = match strides {
        Option::Some(strides) => strides,
        Option::None => {
            let mut strides: Array<usize> = array![];
            let mut i = 2;
            while i != (*X).shape.len() {
                strides.append(1);
                i += 1;
            };

            strides.span()
        },
    };

    let group = match group {
        Option::Some(group) => group,
        Option::None => { 1 },
    };
    let auto_pad = match auto_pad {
        Option::Some(auto_pad) => auto_pad,
        Option::None => { AUTO_PAD::NOTSET },
    };

    if group > 1 {
        let sN = *(*X).shape.at(0);

        let mut res_b: Array<usize> = array![];
        let mut res_cv = array![];

        let mut td = 0;
        let mg = *(*W).shape.at(0) / group;
        let dw = *(*W).shape.at(1);

        let X_stride = stride((*X).shape);
        let mut gx_shape = array![1, dw];
        let mut i = 2;
        while i != (*X).shape.len() {
            gx_shape.append(*(*X).shape.at(i));
            i += 1;
        };

        let gx_shape = gx_shape.span();

        let W_stride = stride((*W).shape);
        let mut gw_shape = array![mg];
        let mut i = 1;
        while i != (*W).shape.len() {
            gw_shape.append(*(*W).shape.at(i));
            i += 1;
        };

        let gw_shape = gw_shape.span();

        let mut b = 0;
        while b != sN {
            let mut g = 0;
            while g != group {
                let gx = TensorTrait::new(
                    gx_shape,
                    SpanTrait::slice(
                        (*X).data,
                        b * *X_stride.at(0) + (g * dw) * *X_stride.at(1),
                        *X_stride.at(1) * dw
                    )
                );
                let gw = TensorTrait::new(
                    gw_shape,
                    SpanTrait::slice((*W).data, (g * mg) * *W_stride.at(0), *W_stride.at(0) * mg)
                );
                let cv = conv(
                    @gx,
                    @gw,
                    Option::None,
                    Option::Some(auto_pad),
                    Option::Some(dilations),
                    Option::Some(1),
                    Option::Some(kernel_shape),
                    Option::Some(pads),
                    Option::Some(strides)
                );

                if b == 0 {
                    td += *cv.shape.at(1);
                }

                res_b.append(b);
                res_cv.append(cv);
                g += 1;
            };

            b += 1;
        };

        let res_b = res_b.span();
        let res_cv = res_cv.span();

        let mut final_shape = array![sN, td];

        let mut cv = *res_cv.at(0);

        let mut i = 2;
        while i != cv.shape.len() {
            final_shape.append(*cv.shape.at(i));
            i += 1;
        };

        let final_shape = final_shape.span();

        let mut final: Array<T> = array![];

        let mut p = 0;
        let mut i = 0;

        while i != res_b
            .len() {
                let cv = *res_cv.at(i);

                let mut n = 0;
                while n != cv.data.len() {
                    final.append(*cv.data.at(n));
                    n += 1;
                };

                p += *cv.shape.at(1);
                if p >= td {
                    p = 0;
                }

                i += 1;
            };

        let final = final.span();

        let final = match B {
            Option::Some(B) => {
                let mut final_b: Array<T> = array![];
                let final_stride = stride(final_shape);
                let mut i = 0;
                while i != *final_shape
                    .at(0) {
                        let mut j = 0;
                        while j != B
                            .len() {
                                let mut k = 0;
                                while k != *final_stride
                                    .at(1) {
                                        final_b
                                            .append(
                                                *final
                                                    .at(
                                                        i * *final_stride.at(0)
                                                            + j * *final_stride.at(1)
                                                            + k
                                                    )
                                                    + *B.at(j)
                                            );
                                        k += 1;
                                    };

                                j += 1;
                            };

                        i += 1;
                    };

                final_b.span()
            },
            Option::None => { final },
        };

        return TensorTrait::new(final_shape, final);
    }

    // group == 1
    if *dilations.at(0) != 1 || min(dilations.clone()) != max(dilations.clone()) {
        // computation of the dilated kernel
        let nd = dilations.len();
        let mut new_kernel_shape: Array<usize> = array![];
        let mut new_shape: Array<usize> = array![];
        new_shape.append_span(SpanTrait::slice((*W).shape, 0, (*W).shape.len() - nd));

        let mut i = 0;
        while i != dilations
            .len() {
                let d = *dilations.at(i);
                let di = (*W).shape.len() - nd + i;
                new_shape.append(*(*W).shape.at(di) + (*(*W).shape.at(di) - 1) * (d - 1));
                new_kernel_shape.append(*kernel_shape.at(i) + (*kernel_shape.at(i) - 1) * (d - 1));
                i += 1;
            };

        let new_shape = new_shape.span();
        let new_w_strides = stride(new_shape);

        let mut new_w = NullableVecImpl::new();
        new_w.set(*new_shape.at(0) * *new_w_strides.at(0) - 1, NumberTrait::zero());

        let mut indices = array![];

        indices.append(arange(0, *new_shape.at(0), 1));
        indices.append(arange(0, *new_shape.at(1), 1));

        let mut i = 0;
        while i != dilations
            .len() {
                let d = *dilations.at(i);
                let di = (*W).shape.len() - nd + i;
                indices.append(arange(0, *new_shape.at(di), d));
                i += 1;
            };

        let set_of_all_indices = cartesian(indices.span());

        let mut new_w_arr: Array<T> = array![];

        let mut i = 0;
        let mut prev = 0;
        while i != (*W)
            .data
            .len() {
                let nd_index = *set_of_all_indices.at(i);
                let mut flatten_index = 0;
                let mut j = 0;
                while j != nd_index
                    .len() {
                        flatten_index += *nd_index.at(j) * *new_w_strides.at(j);
                        j += 1;
                    };

                if flatten_index > prev + 1 {
                    let mut j = prev + 1;
                    while j != flatten_index {
                        new_w_arr.append(NumberTrait::zero());
                    };

                    j += 1;
                }

                new_w_arr.append(*(*W).data.at(i));
                new_w.set(flatten_index, *(*W).data.at(i));
                prev = flatten_index;
                i += 1;
            };
    }

    let pads = match auto_pad {
        AUTO_PAD::NOTSET => { pads },
        AUTO_PAD::SAME_UPPER => {
            let mut head: Array<usize> = array![];
            let mut tail: Array<usize> = array![];
            let mut i = 0;
            while i != nd {
                let d = *(*X).shape.at(i);
                let target_size = (d + *strides.at(i) - 1) / *strides.at(i);
                let pad_needed = (target_size - 1) * *strides.at(i) + *kernel_shape.at(i) - d;
                let pad_head = pad_needed / 2;
                let pad_tail = pad_needed - pad_head;
                head.append(pad_head);
                tail.append(pad_tail);
                i += 1;
            };

            head.append_span(tail.span());
            let pads = head.span();
            pads
        },
        AUTO_PAD::SAME_LOWER => {
            let mut head: Array<usize> = array![];
            let mut tail: Array<usize> = array![];
            let mut i = 0;
            while i != nd {
                let d = *(*X).shape.at(i);
                let target_size = (d + *strides.at(i) - 1) / *strides.at(i);
                let pad_needed = (target_size - 1) * *strides.at(i) + *kernel_shape.at(i) - d;
                let pad_head = (pad_needed + 1) / 2;
                let pad_tail = pad_needed - pad_head;
                head.append(pad_head);
                tail.append(pad_tail);
                i += 1;
            };

            head.append_span(tail.span());
            let pads = head.span();
            pads
        },
        AUTO_PAD::VALID => {
            let mut head: Array<usize> = array![];
            let mut tail: Array<usize> = array![];
            let mut i = 0;
            while i != nd {
                let d = *(*X).shape.at(i);
                let target_size = (d + *strides.at(i) - 1) / *strides.at(i);
                let pad_needed = (target_size - 1) * *strides.at(i) + *kernel_shape.at(i) - d;
                let pad_head = pad_needed / 2;
                let pad_tail = pad_needed - pad_head;
                head.append(pad_head);
                tail.append(pad_tail);
                i += 1;
            };

            head.append_span(tail.span());
            let pads = head.span();
            pads
        },
    };

    if (*X).shape.len() == 3 {
        let sN = *(*X).shape.at(0);
        let sC = *(*X).shape.at(1);
        let sH = *(*X).shape.at(2);

        let sM = *(*W).shape.at(0);

        let kh = *kernel_shape.at(0);
        let sth = *strides.at(0);

        let h_out = ((sH - kh + *pads.at(0) + *pads.at(1)) / sth) + 1;

        let h0 = *pads.at(0);
        let oh: i32 = -1 * (kh % 2).into();
        let bh: i32 = -h0.into();
        let eh = h_out * sth;
        let mut res = NullableVecImpl::new();

        let res_shape = array![sN, sM, h_out].span();
        let res_strides = stride(res_shape);
        res.set(sN * *res_strides.at(0) - 1, NumberTrait::zero());

        match B {
            Option::Some(B) => {
                let mut i = 0;
                while i != sN {
                    let mut j = 0;
                    while j != sM {
                        let b_j = *B.at(j);
                        let mut k = 0;
                        while k != h_out {
                            res.set(i * *res_strides.at(0) + j * *res_strides.at(1) + k, b_j);
                            k += 1;
                        };

                        j += 1;
                    };

                    i += 1;
                };
            },
            Option::None => {},
        }

        let mut n = 0;
        while n != sN {
            let mut nw = 0;
            while nw != sM {
                let mut c = 0;
                while c != sC {
                    let w = SpanTrait::slice((*W).data, nw * sC * kh + c * kh, kh);

                    let mut io = bh;
                    while io < eh
                        .into() {
                            let hr = (io - bh) / sth.into();
                            if hr < h_out.into() {
                                let i = io + (kh % 2).into();

                                let ih1 = I32Number::max(0, i + oh).into();
                                let ih2 = I32Number::min(i + oh + kh.into(), sH.into()).into();
                                let img = SpanTrait::slice(
                                    (*X).data, n * sN + c * sC + ih1, ih2 - ih1
                                );

                                let s = if w.len() != img.len() {
                                    let jh1 = I32Number::max(0, -i - oh).into();
                                    let jh2 = I32Number::min(sH.into() - (i + oh), kh.into())
                                        .into();

                                    let w_ = SpanTrait::slice(w, jh1, jh2 - jh1);
                                    assert(w_.len() == img.len(), 'unexpected w and img len');
                                    dot(img, w_)
                                } else {
                                    dot(img, w)
                                };

                                let hr = if hr < 0 {
                                    *res_strides.at(1) - hr.into()
                                } else {
                                    hr.into()
                                };

                                res
                                    .set(
                                        n * *res_strides.at(0) + nw * *res_strides.at(1) + hr,
                                        res
                                            .at(
                                                n * *res_strides.at(0)
                                                    + nw * *res_strides.at(1)
                                                    + hr
                                            )
                                            + s
                                    );
                            }

                            io += sth.into();
                        };

                    c += 1;
                };

                nw += 1;
            };

            n += 1;
        };

        let mut res_data: Array<T> = array![];
        let mut i = 0;
        while i != res.len() {
            res_data.append(res.at(i));
            i += 1;
        };

        return TensorTrait::new(res_shape, res_data.span());
    }

    if (*X).shape.len() == 4 {
        let sN = *(*X).shape.at(0);
        let sC = *(*X).shape.at(1);
        let sH = *(*X).shape.at(2);
        let sW = *(*X).shape.at(3);

        let sM = *(*W).shape.at(0);

        let kh = *kernel_shape.at(0);
        let kw = *kernel_shape.at(1);

        let sth = *strides.at(0);
        let stw = *strides.at(1);

        let h_out = ((sH - kh + *pads.at(0) + *pads.at(2)) / sth) + 1;
        let w_out = ((sW - kw + *pads.at(1) + *pads.at(3)) / stw) + 1;

        let h0 = *pads.at(0);
        let w0 = *pads.at(1);

        let oh: i32 = -1 * (kh % 2).into();
        let ow: i32 = -1 * (kw % 2).into();
        let bh: i32 = -h0.into();
        let bw: i32 = -w0.into();
        let eh = h_out * sth;
        let ew = w_out * stw;

        let mut res = NullableVecImpl::new();
        let res_shape = array![sN, sM, h_out, w_out].span();
        let res_strides = stride(res_shape);
        res.set(sN * *res_strides.at(0) - 1, NumberTrait::zero());

        match B {
            Option::Some(B) => {
                let mut i = 0;
                while i != sN {
                    let mut j = 0;
                    while j != sM {
                        let b_j = *B.at(j);
                        let mut k = 0;
                        while k != h_out {
                            let mut l = 0;
                            while l != w_out {
                                res
                                    .set(
                                        i * *res_strides.at(0)
                                            + j * *res_strides.at(1)
                                            + k * *res_strides.at(2)
                                            + l,
                                        b_j
                                    );
                                l += 1;
                            };

                            k += 1;
                        };

                        j += 1;
                    };

                    i += 1;
                };
            },
            Option::None => {},
        }

        let mut n = 0;
        while n != sN {
            let mut nw = 0;
            while nw != sM {
                let mut c = 0;
                while c != sC {
                    let w = SpanTrait::slice(
                        (*W).data, nw * (sC * kh * kw) + c * (kh * kw), kh * kw
                    );

                    let mut io = bh;
                    while io < eh
                        .into() {
                            let hr = (io - bh) / sth.into();
                            if hr < h_out.into() {
                                let i = io + (kh % 2).into();
                                let ih1 = I32Number::max(0, i + oh).into();
                                let ih2 = I32Number::min(i + oh + kh.into(), sH.into()).into();

                                let mut jo = bw;
                                while jo < ew
                                    .into() {
                                        let wr = (jo - bw) / stw.into();
                                        if wr < w_out.into() {
                                            let j = jo + (kw % 2).into();
                                            let iw1 = I32Number::max(0, j + ow).into();
                                            let iw2 = I32Number::min(j + ow + kw.into(), sW.into())
                                                .into();

                                            let mut img: Array<T> = array![];
                                            let mut ihi = ih1;
                                            while ihi != ih2 {
                                                img
                                                    .append_span(
                                                        SpanTrait::slice(
                                                            (*X).data,
                                                            n * (sC * sH * sW)
                                                                + c * (sH * sW)
                                                                + ihi * sW
                                                                + iw1,
                                                            iw2 - iw1
                                                        )
                                                    );
                                                ihi += 1;
                                            };

                                            let img = img.span();

                                            let s = if w.len() != img.len() {
                                                let jh1 = I32Number::max(0, -i - oh).into();
                                                let jh2 = I32Number::min(
                                                    sH.into() - (i + oh), kh.into()
                                                )
                                                    .into();

                                                let jw1 = I32Number::max(0, -j - ow).into();
                                                let jw2 = I32Number::min(
                                                    sW.into() - (j + ow), kw.into()
                                                )
                                                    .into();

                                                let mut w_: Array<T> = array![];
                                                let mut jhj = jh1;
                                                while jhj != jh2 {
                                                    w_
                                                        .append_span(
                                                            SpanTrait::slice(
                                                                w, jhj * kw + jw1, jw2 - jw1
                                                            )
                                                        );
                                                    jhj += 1;
                                                };

                                                let w_ = w_.span();

                                                assert(
                                                    w_.len() == img.len(),
                                                    'unexpected w and img len'
                                                );
                                                dot(img, w_)
                                            } else {
                                                dot(img, w)
                                            };

                                            let hr = if hr < 0 {
                                                h_out - hr.into()
                                            } else {
                                                hr.into()
                                            };

                                            let wr = if wr < 0 {
                                                w_out - wr.into()
                                            } else {
                                                wr.into()
                                            };

                                            res
                                                .set(
                                                    n * *res_strides.at(0)
                                                        + nw * *res_strides.at(1)
                                                        + hr * *res_strides.at(2)
                                                        + wr,
                                                    res
                                                        .at(
                                                            n * *res_strides.at(0)
                                                                + nw * *res_strides.at(1)
                                                                + hr * *res_strides.at(2)
                                                                + wr
                                                        )
                                                        + s
                                                );
                                        }

                                        jo += stw.into();
                                    };
                            }

                            io += sth.into();
                        };

                    c += 1;
                };

                nw += 1;
            };

            n += 1;
        };

        let mut res_data: Array<T> = array![];
        let mut i = 0;
        while i != res.len() {
            res_data.append(res.at(i));
            i += 1;
        };

        return TensorTrait::new(res_shape, res_data.span());
    }

    if (*X).shape.len() == 5 {
        let sN = *(*X).shape.at(0);
        let sC = *(*X).shape.at(1);
        let sH = *(*X).shape.at(2);
        let sW = *(*X).shape.at(3);
        let sZ = *(*X).shape.at(4);

        let sM = *(*W).shape.at(0);

        let kh = *kernel_shape.at(0);
        let kw = *kernel_shape.at(1);
        let kz = *kernel_shape.at(2);

        let sth = *strides.at(0);
        let stw = *strides.at(1);
        let stz = *strides.at(2);

        let h_out = ((sH - kh + *pads.at(0) + *pads.at(3)) / sth) + 1;
        let w_out = ((sW - kw + *pads.at(1) + *pads.at(4)) / stw) + 1;
        let z_out = ((sZ - kz + *pads.at(2) + *pads.at(5)) / stz) + 1;

        let h0 = *pads.at(0);
        let w0 = *pads.at(1);
        let z0 = *pads.at(2);

        let oh: i32 = -1 * (kh % 2).into();
        let ow: i32 = -1 * (kw % 2).into();
        let oz: i32 = -1 * (kz % 2).into();

        let bh: i32 = -h0.into();
        let bw: i32 = -w0.into();
        let bz: i32 = -z0.into();

        let eh = h_out * sth;
        let ew = w_out * stw;
        let ez = z_out * stz;

        let mut res = NullableVecImpl::new();
        let res_shape = array![sN, sM, h_out, w_out, z_out].span();
        let res_strides = stride(res_shape);
        res.set(sN * *res_strides.at(0) - 1, NumberTrait::zero());

        match B {
            Option::Some(B) => {
                let mut i = 0;
                while i != sN {
                    let mut j = 0;
                    while j != sM {
                        let b_j = *B.at(j);
                        let mut k = 0;
                        while k != h_out {
                            let mut l = 0;
                            while l != w_out {
                                let mut m = 0;
                                while m != z_out {
                                    res
                                        .set(
                                            i * *res_strides.at(0)
                                                + j * *res_strides.at(1)
                                                + k * *res_strides.at(2)
                                                + l * *res_strides.at(3)
                                                + m,
                                            b_j
                                        );
                                    m += 1;
                                };

                                l += 1;
                            };

                            k += 1;
                        };

                        j += 1;
                    };

                    i += 1;
                };
            },
            Option::None => {},
        }

        let mut n = 0;
        while n != sN {
            let mut nw = 0;
            while nw != sM {
                let mut c = 0;
                while c != sC {
                    let w = SpanTrait::slice(
                        (*W).data, nw * (sC * kh * kw * kz) + c * (kh * kw * kz), kh * kw * kz
                    );

                    let mut io = bh;
                    while io < eh
                        .into() {
                            let hr = (io - bh) / sth.into();
                            if hr < h_out.into() {
                                let i = io + (kh % 2).into();
                                let ih1 = I32Number::max(0, i + oh).into();
                                let ih2 = I32Number::min(i + oh + kh.into(), sH.into()).into();

                                let mut jo = bw;
                                while jo < ew
                                    .into() {
                                        let wr = (jo - bw) / stw.into();
                                        if wr < w_out.into() {
                                            let j = jo + (kw % 2).into();
                                            let iw1 = I32Number::max(0, j + ow).into();
                                            let iw2 = I32Number::min(j + ow + kw.into(), sW.into())
                                                .into();

                                            let mut zo = bz;
                                            while zo < ez
                                                .into() {
                                                    let zr = (zo - bz) / stz.into();
                                                    if zr < z_out.into() {
                                                        let z = zo + (kz % 2).into();
                                                        let iz1 = I32Number::max(0, z + oz).into();
                                                        let iz2 = I32Number::min(
                                                            z + oz + kz.into(), sW.into()
                                                        )
                                                            .into();

                                                        let mut img: Array<T> = array![];
                                                        let mut ihi = ih1;
                                                        while ihi != ih2 {
                                                            let mut iwi = iw1;
                                                            while iwi != iw2 {
                                                                img
                                                                    .append_span(
                                                                        SpanTrait::slice(
                                                                            (*X).data,
                                                                            n * (sC * sH * sW * sZ)
                                                                                + c * (sH * sW * sZ)
                                                                                + ihi * (sW * sZ)
                                                                                + iwi * sZ
                                                                                + iz1,
                                                                            iz2 - iz1
                                                                        )
                                                                    );
                                                                iwi += 1;
                                                            };

                                                            ihi += 1;
                                                        };

                                                        let img = img.span();

                                                        let s = if w.len() != img.len() {
                                                            let jh1 = I32Number::max(0, -i - oh)
                                                                .into();
                                                            let jh2 = I32Number::min(
                                                                sH.into() - (i + oh), kh.into()
                                                            )
                                                                .into();

                                                            let jw1 = I32Number::max(0, -j - ow)
                                                                .into();
                                                            let jw2 = I32Number::min(
                                                                sW.into() - (j + ow), kw.into()
                                                            )
                                                                .into();

                                                            let jz1 = I32Number::max(0, -z - oz)
                                                                .into();
                                                            let jz2 = I32Number::min(
                                                                sZ.into() - (z + oz), kz.into()
                                                            )
                                                                .into();

                                                            let mut w_: Array<T> = array![];
                                                            let mut jhj = jh1;
                                                            while jhj != jh2 {
                                                                let mut jwj = jw1;
                                                                while jwj != jw2 {
                                                                    w_
                                                                        .append_span(
                                                                            SpanTrait::slice(
                                                                                w,
                                                                                jhj * kw * kz
                                                                                    + jwj * kz
                                                                                    + jz1,
                                                                                jz2 - jz1
                                                                            )
                                                                        );
                                                                    jwj += 1;
                                                                };

                                                                jhj += 1;
                                                            };

                                                            let w_ = w_.span();

                                                            assert(
                                                                w_.len() == img.len(),
                                                                'unexpected w and img len'
                                                            );
                                                            dot(img, w_)
                                                        } else {
                                                            dot(img, w)
                                                        };

                                                        let hr = if hr < 0 {
                                                            h_out - hr.into()
                                                        } else {
                                                            hr.into()
                                                        };

                                                        let wr = if wr < 0 {
                                                            w_out - wr.into()
                                                        } else {
                                                            wr.into()
                                                        };

                                                        let zr = if zr < 0 {
                                                            z_out - zr.into()
                                                        } else {
                                                            zr.into()
                                                        };

                                                        res
                                                            .set(
                                                                n * *res_strides.at(0)
                                                                    + nw * *res_strides.at(1)
                                                                    + hr * *res_strides.at(2)
                                                                    + wr * *res_strides.at(3)
                                                                    + zr,
                                                                res
                                                                    .at(
                                                                        n * *res_strides.at(0)
                                                                            + nw
                                                                                * *res_strides.at(1)
                                                                            + hr
                                                                                * *res_strides.at(2)
                                                                            + wr
                                                                                * *res_strides.at(3)
                                                                            + zr
                                                                    )
                                                                    + s
                                                            );
                                                    }

                                                    zo += stz.into();
                                                };
                                        }

                                        jo += stw.into();
                                    };
                            }

                            io += sth.into();
                        };

                    c += 1;
                };

                nw += 1;
            };

            n += 1;
        };

        let mut res_data: Array<T> = array![];
        let mut i = 0;
        while i != res.len() {
            res_data.append(res.at(i));
            i += 1;
        };

        return TensorTrait::new(res_shape, res_data.span());
    }

    // if (*X).shape.len() > 5

    let sN = *(*X).shape.at(0);
    let sC = *(*X).shape.at(1);

    let sM = *(*W).shape.at(0);

    let w_stride = stride((*W).shape);
    let x_stride = stride((*X).shape);

    let mut shape_out: Array<usize> = array![];
    let mut o_index: Array<i32> = array![];
    let mut b_index: Array<i32> = array![];
    let mut e_index: Array<usize> = array![];

    let mut range_len: Array<usize> = array![];

    let mut i = 0;
    while i != nd {
        shape_out
            .append(
                ((*(*X).shape.at(2 + i) - *kernel_shape.at(i) + *pads.at(i) + *pads.at(i + nd))
                    / *strides.at(i))
                    + 1
            );
        let k = *kernel_shape.at(i);
        o_index.append(-1 * (k % 2).into());
        b_index.append(-(*pads.at(i)).into());
        e_index.append(*shape_out.at(i) * *strides.at(i));
        range_len.append((((*e_index.at(i)).into() - *b_index.at(i)).into()) / *strides.at(i));
        i += 1;
    };

    let o_index = o_index.span();
    let b_index = b_index.span();

    let shape_out = shape_out.span();

    let range_len = range_len.span();
    let range_stride = stride(range_len);

    let mut res_shape = array![sN, sM];
    res_shape.append_span(shape_out);
    let res_shape = res_shape.span();

    let res_strides = stride(res_shape);

    let mut res = NullableVecImpl::new();
    res.set(sN * *res_strides.at(0) - 1, NumberTrait::zero());

    match B {
        Option::Some(B) => {
            let mut i = 0;
            while i != sN {
                let mut j = 0;
                while j != sM {
                    let b_j = *B.at(j);
                    let mut k = 0;
                    while k != *res_strides
                        .at(1) {
                            res.set(i * *res_strides.at(0) + j * *res_strides.at(1) + k, b_j);
                            k += 1;
                        };

                    j += 1;
                };

                i += 1;
            };
        },
        Option::None => {},
    }

    let mut n = 0;
    while n != sN {
        let mut nw = 0;
        while nw != sM {
            let mut c = 0;
            while c != sC {
                let w = SpanTrait::slice(
                    (*W).data, nw * *w_stride.at(0) + c * *w_stride.at(1), *w_stride.at(1)
                );
                let mut i = 0;
                while i != *range_len.at(0)
                    * *range_stride
                        .at(0) {
                            let mut io_index: Array<i32> = array![];
                            let mut r_index: Array<i32> = array![];
                            let mut flatten_index = i;

                            let mut nx = 0;
                            while nx != nd {
                                let (n_index, rem) = DivRem::div_rem(
                                    flatten_index, (*range_stride.at(nx)).try_into().unwrap()
                                );

                                flatten_index = rem;
                                io_index
                                    .append(
                                        n_index.into() * (*strides.at(nx)).into() + *b_index.at(nx)
                                    );
                                r_index.append(n_index.into());
                                nx += 1;
                            };

                            if r_index_check(r_index.span(), shape_out) {
                                let mut indices: Array<i32> = array![];
                                let mut i1_index: Array<usize> = array![];
                                let mut i2_index: Array<usize> = array![];
                                let mut idiff_index: Array<usize> = array![];

                                let mut nx = 0;
                                while nx != nd {
                                    indices
                                        .append(
                                            *io_index.at(nx) + (*kernel_shape.at(nx) % 2).into()
                                        );
                                    i1_index
                                        .append(
                                            I32Number::max(0, *indices.at(nx) + *o_index.at(nx))
                                                .into()
                                        );
                                    i2_index
                                        .append(
                                            I32Number::min(
                                                (*(*X).shape.at(nx + 2)).into(),
                                                *indices.at(nx)
                                                    + *o_index.at(nx)
                                                    + (*kernel_shape.at(nx)).into()
                                            )
                                                .into()
                                        );

                                    if nx != nd - 1 {
                                        idiff_index.append(*i2_index.at(nx) - *i1_index.at(nx));
                                    }
                                    nx += 1;
                                };

                                let i1_index = i1_index.span();
                                let mut img: Array<T> = array![];

                                let img = if nx == 1 {
                                    let img = SpanTrait::slice(
                                        (*X).data,
                                        n * sN + c * sC + *i1_index.at(nd - 1),
                                        *i2_index.at(nd - 1) - *i1_index.at(nd - 1)
                                    );
                                    img
                                } else {
                                    let i_stride = stride(idiff_index.span());

                                    let mut ii = 0;
                                    while ii != *i_stride.at(0)
                                        * *idiff_index
                                            .at(0) {
                                                let mut flatten_index = ii;
                                                let mut start = n * *x_stride.at(0)
                                                    + c * *x_stride.at(1);

                                                let mut nx = 0;
                                                while nx != nd
                                                    - 1 {
                                                        let (ii_index, rem) = DivRem::div_rem(
                                                            flatten_index,
                                                            (*i_stride.at(nx)).try_into().unwrap()
                                                        );
                                                        flatten_index = rem;

                                                        start += (*i1_index.at(nx) + ii_index)
                                                            * *x_stride.at(2 + nx);
                                                        nx += 1;
                                                    };

                                                img
                                                    .append_span(
                                                        SpanTrait::slice(
                                                            (*X).data,
                                                            start + *i1_index.at(nd - 1),
                                                            *i2_index.at(nd - 1)
                                                                - *i1_index.at(nd - 1)
                                                        )
                                                    );
                                                ii += 1;
                                            };

                                    img.span()
                                };

                                let s = if w.len() != img.len() {
                                    let mut j1_index: Array<usize> = array![];
                                    let mut j2_index: Array<usize> = array![];
                                    let mut jdiff_index: Array<usize> = array![];

                                    let mut nx = 0;
                                    while nx != nd {
                                        j1_index
                                            .append(
                                                I32Number::max(
                                                    0, -*indices.at(nx) - *o_index.at(nx)
                                                )
                                                    .into()
                                            );
                                        j2_index
                                            .append(
                                                I32Number::min(
                                                    (*(*X).shape.at(nx + 2)).into()
                                                        - *indices.at(nx)
                                                        - *o_index.at(nx),
                                                    (*kernel_shape.at(nx)).into()
                                                )
                                                    .into()
                                            );
                                        if nx != nd - 1 {
                                            jdiff_index.append(*j2_index.at(nx) - *j1_index.at(nx));
                                        }
                                        nx += 1;
                                    };

                                    let j1_index = j1_index.span();

                                    let mut w_: Array<T> = array![];

                                    let w_ = if nx == 1 {
                                        let w_ = SpanTrait::slice(
                                            w,
                                            *j1_index.at(nd - 1),
                                            *j2_index.at(nd - 1) - *j1_index.at(nd - 1)
                                        );
                                        w_
                                    } else {
                                        let j_stride = stride(jdiff_index.span());

                                        let mut jj = 0;
                                        while jj != *j_stride.at(0)
                                            * *jdiff_index
                                                .at(0) {
                                                    let mut flatten_index = jj;
                                                    let mut start = 0;

                                                    let mut nx = 0;
                                                    while nx != nd
                                                        - 1 {
                                                            let (jj_index, rem) = DivRem::div_rem(
                                                                flatten_index,
                                                                (*j_stride.at(nx))
                                                                    .try_into()
                                                                    .unwrap()
                                                            );
                                                            flatten_index = rem;
                                                            start += (*j1_index.at(nx) + jj_index)
                                                                * *kernel_shape.at(nx);
                                                            nx += 1;
                                                        };
                                                    w_
                                                        .append_span(
                                                            SpanTrait::slice(
                                                                w,
                                                                start + *j1_index.at(nd - 1),
                                                                *j2_index.at(nd - 1)
                                                                    - *j1_index.at(nd - 1)
                                                            )
                                                        );
                                                    jj += 1;
                                                };

                                        w_.span()
                                    };

                                    dot(img, w_)
                                } else {
                                    dot(img, w)
                                };

                                let mut res_index = n * *res_strides.at(0)
                                    + nw * *res_strides.at(1);

                                let mut nx = 0;
                                while nx != nd {
                                    res_index += (*r_index.at(nx)).into() * *res_strides.at(2 + nx);
                                    nx += 1;
                                };

                                res.set(res_index, res.at(res_index) + s);
                            };

                            i += 1
                        };

                c += 1;
            };

            nw += 1;
        };

        n += 1;
    };

    let mut res_data: Array<T> = array![];
    let mut i = 0;
    while i != res.len() {
        res_data.append(res.at(i));
        i += 1;
    };

    TensorTrait::new(res_shape, res_data.span())
}

fn r_index_check(r_index: Span<i32>, shape_out: Span<usize>) -> bool {
    let mut i = 0;
    let flag = loop {
        if i == r_index.len() {
            break true;
        }
        if *r_index.at(i) >= (*shape_out.at(i)).into() {
            break false;
        }
        i += 1;
    };

    flag
}

fn prod<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +Mul<T>,>(
    pA: Span<T>, start: usize
) -> T {
    let mut i = start;
    let mut prod = NumberTrait::one();
    while i != pA.len() {
        prod = prod * (*pA.at(i));
        i += 1;
    };

    prod
}

fn min(mut a: Span<usize>) -> usize {
    assert(a.len() > 0, 'span cannot be empty');

    let mut min = *a.at(0);
    loop {
        match a.pop_front() {
            Option::Some(v) => { if *v < min {
                min = *v;
            }; },
            Option::None => { break min; }
        };
    }
}

fn max(mut a: Span<usize>) -> usize {
    assert(a.len() > 0, 'span cannot be empty');

    let mut max = *a.at(0);
    loop {
        match a.pop_front() {
            Option::Some(v) => { if *v > max {
                max = *v;
            }; },
            Option::None => { break max; }
        };
    }
}

fn arange(start: usize, end: usize, step: usize) -> Span<usize> {
    assert((end - start) % step == 0, 'incompatible step value');

    let mut arr: Array<usize> = array![];
    let mut i = start;
    while i < end {
        arr.append(i);
        i += step;
    };

    arr.span()
}


fn cartesian(mut arrays: Span<Span<usize>>,) -> Span<Span<usize>> {
    let mut n = 1;
    let mut i = arrays.len() - 1;
    loop {
        n = n * (*(arrays.at(i))).len();
        if i == 0 {
            break;
        }
        i -= 1;
    };

    let mut i = 0;
    let mut size_arrays: Array<usize> = array![];
    while i != arrays.len() {
        size_arrays.append((*(arrays.at(i))).len());
        i += 1;
    };

    let size_arrays = size_arrays.span();
    let mut output_arrays = array![];
    let mut m = n;

    let mut i = 0;
    while i != arrays
        .len() {
            m = m / (*(arrays.at(i))).len();
            let mut out = repeat(*(arrays.at(i)), m);
            out = repeat_2(out, size_arrays, i);

            output_arrays.append(out);
            i += 1;
        };

    let output_arrays = output_arrays.span();

    let mut i = 0;
    let mut ret = ArrayTrait::new();
    while i != n {
        let mut j = 0;
        let mut x: Array<usize> = array![];
        while j != arrays.len() {
            x.append(*(output_arrays.at(j)).at(i));
            j += 1;
        };

        ret.append(x.span());
        i += 1;
    };

    ret.span()
}

fn repeat_2(mut array: Array<usize>, size_array: Span<usize>, index: usize) -> Array<usize> {
    let mut size = array.len();
    let mut i = 0;
    while i != index {
        let mut j = 1;
        while j != *size_array
            .at(index - 1 - i) {
                let mut k = 0;
                while k != size {
                    array.append(*array.at(k));
                    k += 1;
                };

                j += 1;
            };

        size = size * *size_array.at(index - 1 - i);
        i += 1;
    };

    array
}

fn repeat(array: Span<usize>, m: usize,) -> Array<usize> {
    let mut out: Array<usize> = array![];
    let mut j = 0;
    while j != array
        .len() {
            let mut k = 0;
            while k != m {
                out.append(*array.at(j));
                k += 1;
            };

            j += 1;
        };

    out
}

fn dot<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Add<T>, +TensorTrait<T>, +AddEq<T>, +Mul<T>,
>(
    a: Span<T>, b: Span<T>
) -> T {
    let mut i = 0;
    let mut sum = NumberTrait::zero();
    while i != a.len() {
        sum = sum + *a.at(i) * *b.at(i);
        i += 1;
    };

    sum
}
