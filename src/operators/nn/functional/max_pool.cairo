use core::clone::Clone;
use core::option::OptionTrait;
use core::array::ArrayTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, I32Tensor};
use orion::operators::vec::{NullableVec, NullableVecImpl};
use orion::operators::tensor::core::{stride};
use core::debug::PrintTrait;
use core::traits::Into;
use orion::numbers::{U32IntoI32, I32IntoU32, I32Div, I32Number};

use orion::operators::nn::functional::common_pool::{common_pool};
use orion::operators::nn::{AUTO_PAD, POOLING_TYPE};

/// Cf: NNTrait::max_pool docstring
fn max_pool<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +Div<T>,
    +AddEq<T>,
    +PrintTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
>(
    X: @Tensor<T>,
    auto_pad: Option<AUTO_PAD>,
    ceil_mode: Option<usize>,
    dilations: Option<Span<usize>>,
    kernel_shape: Span<usize>,
    pads: Option<Span<usize>>,
    storage_order: Option<usize>,
    strides: Option<Span<usize>>,
    output_len: usize,
) -> (Tensor<T>, Option<Tensor<usize>>) {
    match dilations {
        Option::Some(dilations) => {
            if (min(dilations) != max(dilations) || min(dilations) != 1) {
                max_pool_implementation(
                    X,
                    auto_pad,
                    ceil_mode,
                    Option::Some(dilations),
                    kernel_shape,
                    pads,
                    storage_order,
                    strides,
                    output_len,
                )
            } else {
                match strides {
                    Option::Some(strides) => {
                        if (min(strides) != max(strides) || min(strides) != 1) {
                            max_pool_implementation(
                                X,
                                auto_pad,
                                ceil_mode,
                                Option::Some(dilations),
                                kernel_shape,
                                pads,
                                storage_order,
                                Option::Some(strides),
                                output_len,
                            )
                        } else {
                            common_pool(
                                POOLING_TYPE::MAX,
                                0,
                                X,
                                auto_pad,
                                ceil_mode,
                                Option::Some(dilations),
                                kernel_shape,
                                pads,
                                Option::Some(strides),
                                1,
                            )
                        }
                    },
                    Option::None => {
                        common_pool(
                            POOLING_TYPE::MAX,
                            0,
                            X,
                            auto_pad,
                            ceil_mode,
                            Option::Some(dilations),
                            kernel_shape,
                            pads,
                            Option::None,
                            1,
                        )
                    },
                }
            }
        },
        Option::None => {
            match strides {
                Option::Some(strides) => {
                    if (min(strides) != max(strides) || min(strides) != 1) {
                        max_pool_implementation(
                            X,
                            auto_pad,
                            ceil_mode,
                            Option::None,
                            kernel_shape,
                            pads,
                            storage_order,
                            Option::Some(strides),
                            output_len,
                        )
                    } else {
                        common_pool(
                            POOLING_TYPE::MAX,
                            0,
                            X,
                            auto_pad,
                            ceil_mode,
                            Option::None,
                            kernel_shape,
                            pads,
                            Option::Some(strides),
                            1,
                        )
                    }
                },
                Option::None => {
                    common_pool(
                        POOLING_TYPE::MAX,
                        0,
                        X,
                        auto_pad,
                        ceil_mode,
                        Option::None,
                        kernel_shape,
                        pads,
                        Option::None,
                        1,
                    )
                },
            }
        }
    }
}


fn max_pool_implementation<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +Div<T>,
    +AddEq<T>,
    +PrintTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
>(
    X: @Tensor<T>,
    auto_pad: Option<AUTO_PAD>,
    ceil_mode: Option<usize>,
    dilations: Option<Span<usize>>,
    kernel_shape: Span<usize>,
    pads: Option<Span<usize>>,
    storage_order: Option<usize>,
    strides: Option<Span<usize>>,
    output_len: usize,
) -> (Tensor<T>, Option<Tensor<usize>>) {
    assert((*X).shape.len() >= 3, 'X must have at least 3 dim');
    let n_dims = kernel_shape.len();

    let pads = match pads {
        Option::Some(pads) => pads,
        Option::None => {
            let mut pads = ArrayTrait::new();
            let mut i = 0;
            while i != n_dims {
                pads.append(0);
                pads.append(0);
                i += 1;
            };
            pads.span()
        },
    };
    let dilations = match dilations {
        Option::Some(dilations) => dilations,
        Option::None => {
            let mut dilations = ArrayTrait::new();
            let mut i = 0;
            while i != n_dims {
                dilations.append(1);
                i += 1;
            };
            dilations.span()
        },
    };
    let strides = match strides {
        Option::Some(strides) => strides,
        Option::None => {
            let mut strides = ArrayTrait::new();
            let mut i = 0;
            while i != n_dims {
                strides.append(1);
                i += 1;
            };
            strides.span()
        },
    };

    let auto_pad = match auto_pad {
        Option::Some(auto_pad) => auto_pad,
        Option::None => AUTO_PAD::NOTSET,
    };

    let storage_order = match storage_order {
        Option::Some(storage_order) => storage_order,
        Option::None => 0,
    };

    let input_spatial_shape = SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2);

    let ceil_mode = match ceil_mode {
        Option::Some(ceil_mode) => ceil_mode,
        Option::None => 0,
    };

    let output_spatial_shape = if ceil_mode == 1 {
        let mut output_spatial_shape = ArrayTrait::<usize>::new();
        let mut i = 0;
        while i != input_spatial_shape
            .len() {
                let oss: T = NumberTrait::ceil(
                    (NumberTrait::new_unscaled(
                        (*input_spatial_shape.at(i) + *pads.at(i) + *pads.at(i + n_dims)).into(),
                        false
                    )
                        - NumberTrait::new_unscaled(
                            ((*kernel_shape.at(i) - 1) * *dilations.at(i) + 1).into(), false
                        ))
                        / NumberTrait::new_unscaled((*strides.at(i)).into(), false)
                        + NumberTrait::one()
                );

                let need_to_reduce_out_size_in_ceil_mode = (oss.try_into().unwrap() - 1)
                    * *strides.at(i) >= *input_spatial_shape.at(i)
                    + *pads.at(i);
                if need_to_reduce_out_size_in_ceil_mode {
                    output_spatial_shape.append(oss.try_into().unwrap() - 1);
                } else {
                    output_spatial_shape.append(oss.try_into().unwrap());
                };
                i += 1;
            };

        output_spatial_shape.span()
    } else {
        let mut output_spatial_shape = ArrayTrait::<usize>::new();
        let mut i = 0;
        while i != input_spatial_shape
            .len() {
                let oss: T = NumberTrait::floor(
                    (NumberTrait::new_unscaled(
                        (*input_spatial_shape.at(i) + *pads.at(i) + *pads.at(i + n_dims)).into(),
                        false
                    )
                        - NumberTrait::new_unscaled(
                            ((*kernel_shape.at(i) - 1) * *dilations.at(i) + 1).into(), false
                        ))
                        / NumberTrait::new_unscaled((*strides.at(i)).into(), false)
                        + NumberTrait::one()
                );
                output_spatial_shape.append(oss.try_into().unwrap());
                i += 1;
            };
        output_spatial_shape.span()
    };

    let (pads, output_spatial_shape) = match auto_pad {
        AUTO_PAD::NOTSET => { (pads, output_spatial_shape) },
        AUTO_PAD::SAME_UPPER => {
            let mut output_spatial_shape = ArrayTrait::<usize>::new();
            let mut pad_1 = ArrayTrait::new();
            let mut pad_2 = ArrayTrait::new();
            let mut pads = ArrayTrait::new();

            let mut i = 0;
            while i != input_spatial_shape
                .len() {
                    let oss: T = NumberTrait::ceil(
                        NumberTrait::new_unscaled((*input_spatial_shape.at(i)).into(), false)
                            / NumberTrait::new_unscaled((*strides.at(i)).into(), false)
                    );
                    output_spatial_shape.append(oss.try_into().unwrap());

                    let pad_i = (*output_spatial_shape[i] - 1) * *strides[i]
                        + ((*kernel_shape[i] - 1) * *dilations[i] + 1)
                        - *input_spatial_shape[i];

                    pad_1.append(pad_i / 2);
                    pad_2.append(pad_i - (pad_i / 2));

                    i += 1;
                };

            pads.append_span(pad_1.span());
            pads.append_span(pad_2.span());

            (pads.span(), output_spatial_shape.span())
        },
        AUTO_PAD::SAME_LOWER => {
            let mut output_spatial_shape = ArrayTrait::<usize>::new();
            let mut pad_1 = ArrayTrait::new();
            let mut pad_2 = ArrayTrait::new();
            let mut pads = ArrayTrait::new();

            let mut i = 0;
            while i != input_spatial_shape
                .len() {
                    let oss: T = NumberTrait::floor(
                        NumberTrait::new_unscaled((*input_spatial_shape.at(i)).into(), false)
                            / NumberTrait::new_unscaled((*strides.at(i)).into(), false)
                    );
                    output_spatial_shape.append(oss.try_into().unwrap());

                    let pad_i = (*output_spatial_shape[i] - 1) * *strides[i]
                        + ((*kernel_shape[i] - 1) * *dilations[i] + 1)
                        - *input_spatial_shape[i];

                    pad_1.append(pad_i / 2);
                    pad_2.append(pad_i - (pad_i / 2));

                    i += 1;
                };

            pads.append_span(pad_1.span());
            pads.append_span(pad_2.span());

            (pads.span(), output_spatial_shape.span())
        },
        AUTO_PAD::VALID => {
            let mut output_spatial_shape = ArrayTrait::<usize>::new();
            let mut i = 0;
            while i != input_spatial_shape
                .len() {
                    let oss: T = NumberTrait::ceil(
                        (NumberTrait::new_unscaled((*input_spatial_shape.at(i)).into(), false)
                            - NumberTrait::new_unscaled(
                                ((*kernel_shape.at(i) - 1) * *dilations.at(i) + 1).into(), false
                            )
                            + NumberTrait::one())
                            / NumberTrait::new_unscaled((*strides.at(i)).into(), false)
                    );
                    output_spatial_shape.append(oss.try_into().unwrap());

                    i += 1;
                };

            (pads, output_spatial_shape.span())
        },
    };

    let nd = input_spatial_shape.len();
    if nd == 1 {
        return max_pool_1d(
            X,
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            pads,
            storage_order,
            strides,
            output_spatial_shape,
            output_len
        );
    }
    if nd == 2 {
        return max_pool_2d(
            X,
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            pads,
            storage_order,
            strides,
            output_spatial_shape,
            output_len
        );
    }
    if nd == 3 {
        return max_pool_3d(
            X,
            auto_pad,
            ceil_mode,
            dilations,
            kernel_shape,
            pads,
            storage_order,
            strides,
            output_spatial_shape,
            output_len
        );
    }

    return max_pool_nd(
        X,
        auto_pad,
        ceil_mode,
        dilations,
        kernel_shape,
        pads,
        storage_order,
        strides,
        output_spatial_shape,
        output_len
    );
}


fn max_pool_1d<T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>, +PartialOrd<T>,>(
    X: @Tensor<T>,
    auto_pad: AUTO_PAD,
    ceil_mode: usize,
    dilations: Span<usize>,
    kernel_shape: Span<usize>,
    pads: Span<usize>,
    storage_order: usize,
    strides: Span<usize>,
    output_spatial_shape: Span<usize>,
    output_len: usize,
) -> (Tensor<T>, Option<Tensor<usize>>) {
    let mut y_dims = ArrayTrait::new();
    y_dims.append_span(SpanTrait::slice((*X).shape, 0, 2));
    y_dims.append_span(output_spatial_shape);

    let N = *(*X).shape.at(0);
    let C = *(*X).shape.at(1);

    let x_step = *(*X).shape.at(2);
    let y_step = *y_dims.at(2);

    let total_channels = N * C;

    let stride_h = I32Number::new((*strides.at(0)).into(), false);
    let dilation_h = I32Number::new((*dilations.at(0)).into(), false);
    let ks_h = I32Number::new((*kernel_shape.at(0)).into(), false);
    let pad_h = I32Number::new((*pads.at(0)).into(), false);

    let mut Y_data = ArrayTrait::new();
    let mut I_data = ArrayTrait::new();

    let mut c = 0;
    while c != total_channels {
        let x_d = c * x_step;

        let mut ph = 0;
        while ph != y_step {
            let hstart = I32Number::new((ph).into(), false) * stride_h - pad_h;
            let hend = hstart + ks_h * dilation_h;

            let mut h_index = I32Number::new(1, true);
            let mut Yh: T = NumberTrait::min_value();

            let mut h = hstart;
            while h != hend {
                if h >= 0 && h < x_step.into() {
                    if *(*X).data.at(x_d + h.into()) > Yh {
                        h_index = h.into();
                        Yh = (*(*X).data.at(x_d + h.into()));
                    }
                }
                h += dilation_h;
            };

            Y_data.append(Yh);
            I_data.append((c * x_step) + h_index.into());

            ph += 1;
        };
        c += 1;
    };
    if output_len == 1 {
        return (TensorTrait::new(y_dims.span(), Y_data.span()), Option::<Tensor<usize>>::None);
    }
    return (
        TensorTrait::new(y_dims.span(), Y_data.span()),
        Option::Some(TensorTrait::new(y_dims.span(), I_data.span()))
    );
}

fn max_pool_2d<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +PrintTrait<T>
>(
    X: @Tensor<T>,
    auto_pad: AUTO_PAD,
    ceil_mode: usize,
    dilations: Span<usize>,
    kernel_shape: Span<usize>,
    pads: Span<usize>,
    storage_order: usize,
    strides: Span<usize>,
    output_spatial_shape: Span<usize>,
    output_len: usize,
) -> (Tensor<T>, Option<Tensor<usize>>) {
    let mut y_dims = ArrayTrait::new();
    y_dims.append_span(SpanTrait::slice((*X).shape, 0, 2));
    y_dims.append_span(output_spatial_shape);

    let N = *(*X).shape.at(0);
    let C = *(*X).shape.at(1);
    let H = *(*X).shape.at(2);
    let W = *(*X).shape.at(3);

    let pooled_H = *y_dims.at(2);
    let pooled_W = *y_dims.at(3);

    let x_step = H * W;

    let total_channels = N * C;

    let stride_h = I32Number::new((*strides.at(0)).into(), false);
    let stride_w = I32Number::new((*strides.at(1)).into(), false);

    let dilation_h = I32Number::new((*dilations.at(0)).into(), false);
    let dilation_w = I32Number::new((*dilations.at(1)).into(), false);

    let ks_h = I32Number::new((*kernel_shape.at(0)).into(), false);
    let ks_w = I32Number::new((*kernel_shape.at(1)).into(), false);

    let pad_h = I32Number::new((*pads.at(0)).into(), false);
    let pad_w = I32Number::new((*pads.at(1)).into(), false);

    let mut Y_data = ArrayTrait::new();
    let mut I_data = ArrayTrait::new();

    let X_len = (*X).data.len();

    let mut c = 0;
    while c != total_channels {
        let x_d = c * x_step;

        let mut ph = 0;
        while ph != pooled_H {
            let hstart = I32Number::new((ph).into(), false) * stride_h - pad_h;
            let hend = hstart + ks_h * dilation_h;

            let mut pw = 0;
            while pw != pooled_W {
                let wstart = I32Number::new((pw).into(), false) * stride_w - pad_w;
                let wend = wstart + ks_w * dilation_w;

                let mut h_index = I32Number::new(1, true);
                let mut w_index = I32Number::new(1, true);

                let mut Yh: T = NumberTrait::min_value();

                let mut h = hstart;
                while h != hend {
                    if h >= 0 && h < H.into() {
                        let mut w = wstart;
                        while w != wend {
                            if w >= 0 && w < W.into() {
                                let input_index = h * W.into() + w;
                                if input_index >= 0 && input_index < X_len.into() {
                                    if *(*X).data.at(x_d + input_index.into()) > Yh {
                                        h_index = h.into();
                                        w_index = w.into();
                                        Yh = (*(*X).data.at(x_d + input_index.into()));
                                    }
                                }
                            }
                            w += dilation_w;
                        };
                    };
                    h += dilation_h;
                };

                if Yh != NumberTrait::<T>::min_value() {
                    Y_data.append(Yh);
                    if storage_order == 0 {
                        I_data.append((c * x_step) + h_index.into() * W + w_index.into());
                    } else {
                        I_data.append((c * x_step) + h_index.into() + w_index.into() * H);
                    }
                }
                pw += 1;
            };
            ph += 1;
        };
        c += 1;
    };

    if output_len == 1 {
        return (TensorTrait::new(y_dims.span(), Y_data.span()), Option::<Tensor<usize>>::None);
    }
    return (
        TensorTrait::new(y_dims.span(), Y_data.span()),
        Option::Some(TensorTrait::new(y_dims.span(), I_data.span()))
    );
}

fn max_pool_3d<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +PrintTrait<T>
>(
    X: @Tensor<T>,
    auto_pad: AUTO_PAD,
    ceil_mode: usize,
    dilations: Span<usize>,
    kernel_shape: Span<usize>,
    pads: Span<usize>,
    storage_order: usize,
    strides: Span<usize>,
    output_spatial_shape: Span<usize>,
    output_len: usize,
) -> (Tensor<T>, Option<Tensor<usize>>) {
    let mut y_dims = ArrayTrait::new();
    y_dims.append_span(SpanTrait::slice((*X).shape, 0, 2));
    y_dims.append_span(output_spatial_shape);

    let N = *(*X).shape.at(0);
    let C = *(*X).shape.at(1);
    let H = *(*X).shape.at(2);
    let W = *(*X).shape.at(3);
    let D = *(*X).shape.at(4);

    let pooled_H = *y_dims.at(2);
    let pooled_W = *y_dims.at(3);
    let pooled_D = *y_dims.at(4);

    let x_step = H * W * D;

    let total_channels = N * C;

    let stride_h = I32Number::new((*strides.at(0)).into(), false);
    let stride_w = I32Number::new((*strides.at(1)).into(), false);
    let stride_d = I32Number::new((*strides.at(2)).into(), false);

    let dilation_h = I32Number::new((*dilations.at(0)).into(), false);
    let dilation_w = I32Number::new((*dilations.at(1)).into(), false);
    let dilation_d = I32Number::new((*dilations.at(2)).into(), false);

    let ks_h = I32Number::new((*kernel_shape.at(0)).into(), false);
    let ks_w = I32Number::new((*kernel_shape.at(1)).into(), false);
    let ks_d = I32Number::new((*kernel_shape.at(2)).into(), false);

    let pad_h = I32Number::new((*pads.at(0)).into(), false);
    let pad_w = I32Number::new((*pads.at(1)).into(), false);
    let pad_d = I32Number::new((*pads.at(2)).into(), false);

    let mut Y_data = ArrayTrait::new();
    let mut I_data = ArrayTrait::new();

    let X_len = (*X).data.len();

    let mut c = 0;

    while c != total_channels {
        let x_d = c * x_step;

        let mut ph = 0;
        while ph != pooled_H {
            let hstart = I32Number::new((ph).into(), false) * stride_h - pad_h;
            let hend = hstart + ks_h * dilation_h;

            let mut pw = 0;
            while pw != pooled_W {
                let wstart = I32Number::new((pw).into(), false) * stride_w - pad_w;
                let wend = wstart + ks_w * dilation_w;

                let mut pd = 0;
                while pd != pooled_D {
                    let dstart = I32Number::new((pd).into(), false) * stride_d - pad_d;
                    let dend = dstart + ks_d * dilation_d;

                    let mut h_index = I32Number::new(1, true);
                    let mut w_index = I32Number::new(1, true);
                    let mut d_index = I32Number::new(1, true);

                    let mut Yh: T = NumberTrait::min_value();

                    let mut h = hstart;
                    while h != hend {
                        if h >= 0 && h < H.into() {
                            let mut w = wstart;
                            while w != wend {
                                if w >= 0 && w < W.into() {
                                    let mut d = dstart;
                                    while d != dend {
                                        if d >= 0 && d < D.into() {
                                            let input_index = h * W.into() * D.into()
                                                + w * D.into()
                                                + d;
                                            if input_index >= 0 && input_index < X_len.into() {
                                                if *(*X).data.at(x_d + input_index.into()) > Yh {
                                                    h_index = h.into();
                                                    w_index = w.into();
                                                    d_index = d.into();
                                                    Yh = (*(*X).data.at(x_d + input_index.into()));
                                                }
                                            }
                                        }
                                        d += dilation_d;
                                    };
                                };
                                w += dilation_w;
                            };
                        };
                        h += dilation_h;
                    };
                    Y_data.append(Yh);

                    if storage_order == 0 {
                        I_data
                            .append(
                                (c * x_step)
                                    + h_index.into() * W * D
                                    + w_index.into() * D
                                    + d_index.into()
                            );
                    } else {
                        I_data
                            .append(
                                (c * x_step)
                                    + h_index.into()
                                    + w_index.into() * H
                                    + d_index.into() * H * W
                            );
                    }
                    pd += 1;
                };
                pw += 1;
            };
            ph += 1;
        };
        c += 1;
    };

    if output_len == 1 {
        return (TensorTrait::new(y_dims.span(), Y_data.span()), Option::<Tensor<usize>>::None);
    }
    return (
        TensorTrait::new(y_dims.span(), Y_data.span()),
        Option::Some(TensorTrait::new(y_dims.span(), I_data.span()))
    );
}


fn max_pool_nd<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +PrintTrait<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +Div<T>
>(
    X: @Tensor<T>,
    auto_pad: AUTO_PAD,
    ceil_mode: usize,
    dilations: Span<usize>,
    kernel_shape: Span<usize>,
    pads: Span<usize>,
    storage_order: usize,
    strides: Span<usize>,
    output_spatial_shape: Span<usize>,
    output_len: usize,
) -> (Tensor<T>, Option<Tensor<usize>>) {
    let nd = (*X).shape.len() - 2;

    let mut y_dims = ArrayTrait::new();
    y_dims.append_span(SpanTrait::slice((*X).shape, 0, 2));
    y_dims.append_span(output_spatial_shape);

    let N = *(*X).shape.at(0);
    let C = *(*X).shape.at(1);

    let x_stride = stride((*X).shape);
    let y_stride = stride(y_dims.span());

    let i_stride_storage_order_1 = if storage_order == 1 {
        let i_stride_storage_order_1 = reverse_stride(SpanTrait::slice((*X).shape, 2, nd));
        i_stride_storage_order_1
    } else {
        array![].span()
    };

    let x_step = *x_stride.at(1);
    let y_step = *y_stride.at(1);

    let total_channels = N * C;

    let stride_n: Span<i32> = u32_span_into_i32_span(strides);
    let dilation_n: Span<i32> = u32_span_into_i32_span(dilations);
    let ks_n: Span<i32> = u32_span_into_i32_span(kernel_shape);
    let pad_n: Span<i32> = u32_span_into_i32_span(pads);

    let mut Y_data = ArrayTrait::new();
    let mut I_data = ArrayTrait::new();

    let X_len = (*X).data.len();

    let mut c = 0;
    while c != total_channels {
        let x_d = c * x_step;

        let mut p = 0;
        while p != y_step {
            let mut flatten_index = p;

            let mut nstart = ArrayTrait::new();
            let mut nend = ArrayTrait::new();
            let mut nstep = ArrayTrait::<usize>::new();

            let mut n = 0;
            while n != nd {
                let (pn, rem) = DivRem::div_rem(
                    flatten_index, (*y_stride.at(2 + n)).try_into().unwrap()
                );
                flatten_index = rem;

                let ns = pn.into() * *stride_n.at(n) - *pad_n.at(n);
                nstart.append(ns);
                nend.append(ns + *ks_n.at(n) * *dilation_n.at(n));

                let a: T = NumberTrait::new_unscaled(
                    (*kernel_shape.at(n) * *dilations.at(n)).into(), false
                );
                let b: T = NumberTrait::new_unscaled((*dilations.at(n)).into(), false);
                nstep.append(NumberTrait::ceil(a / b).try_into().unwrap());
                n += 1;
            };

            let nstart = nstart.span();
            let nstride = stride(nstep.span());
            let max_iter = *nstep.at(0) * *nstride.at(0);

            let mut n_index = array![I32Number::new(1, true)].span();

            let mut Yh: T = NumberTrait::min_value();

            let mut i = 0;
            while i != max_iter {
                let mut flatten_index = i;
                let mut is_outside = false;
                let mut i_index = ArrayTrait::new();
                let mut input_index = I32Number::zero();

                let mut n = 0;
                while n != nd {
                    let (item, rem) = DivRem::div_rem(
                        flatten_index, (*nstride.at(n)).try_into().unwrap()
                    );
                    flatten_index = rem;

                    let item_ = item.into() * *dilation_n.at(n) + *nstart.at(n);
                    if item_ < 0 || item_ >= (*(*X).shape.at(2 + n)).into() {
                        is_outside = true;
                    };
                    i_index.append(item_);
                    input_index += item_ * (*x_stride.at(2 + n)).into();

                    n += 1;
                };

                if !is_outside {
                    if input_index >= 0 && input_index < X_len.into() {
                        if *(*X).data.at(x_d + input_index.into()) > Yh {
                            n_index = i_index.span().clone();
                            Yh = (*(*X).data.at(x_d + input_index.into()));
                        };
                    };
                };
                i += 1;
            };
            Y_data.append(Yh);

            if storage_order == 0 {
                let mut index = 0;
                let mut n = 0;
                while n != nd {
                    index += *n_index.at(n) * (*x_stride.at(2 + n)).into();
                    n += 1;
                };
                I_data.append((c * x_step) + index.into());
            } else {
                let mut index = 0;
                let mut n = nd;
                while n != 0 {
                    index += *n_index.at(n - 1) * (*i_stride_storage_order_1.at(nd - n)).into();
                    n -= 1;
                };
                I_data.append((c * x_step) + index.into());
            }
            p += 1;
        };
        c += 1;
    };
    if output_len == 1 {
        return (TensorTrait::new(y_dims.span(), Y_data.span()), Option::<Tensor<usize>>::None);
    }
    return (
        TensorTrait::new(y_dims.span(), Y_data.span()),
        Option::Some(TensorTrait::new(y_dims.span(), I_data.span()))
    );
}


fn u32_span_into_i32_span(mut x: Span<usize>) -> Span<i32> {
    let mut res = ArrayTrait::new();

    loop {
        match x.pop_front() {
            Option::Some(v) => { res.append((*v).into()); },
            Option::None => { break res.span(); }
        };
    }
}

fn reverse_stride(mut a: Span<usize>) -> Span<usize> {
    let mut prod = 1;
    let mut arr = ArrayTrait::new();
    loop {
        match a.pop_front() {
            Option::Some(v) => {
                prod *= *v;
                arr.append(prod);
            },
            Option::None => { break arr.span(); }
        };
    }
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
