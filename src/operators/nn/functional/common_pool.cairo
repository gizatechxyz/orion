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
use orion::numbers::FP16x16;
use orion::operators::nn::{AUTO_PAD, POOLING_TYPE};


fn common_pool<
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
    +Into<i32, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
>(
    pooling_type: POOLING_TYPE,
    count_include_pad: usize,
    X: @Tensor<T>,
    auto_pad: Option<AUTO_PAD>,
    ceil_mode: Option<usize>,
    dilations: Option<Span<usize>>,
    kernel_shape: Span<usize>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
    p: usize,
) -> (Tensor<T>, Option<Tensor<i32>>) {
    let padding_value: T = match pooling_type {
        POOLING_TYPE::AVG => {
            let padding_value = if count_include_pad == 0 {
                NumberTrait::min_value()
            } else {
                NumberTrait::zero()
            };
            padding_value
        },
        POOLING_TYPE::LPPOOL => {
            let padding_value = if count_include_pad == 0 {
                NumberTrait::min_value()
            } else {
                NumberTrait::zero()
            };
            padding_value
        },
        POOLING_TYPE::MAX => { NumberTrait::min_value() },
    };

    let ceil_mode = match ceil_mode {
        Option::Some(ceil_mode) => { ceil_mode },
        Option::None => { 0 },
    };

    let auto_pad = match auto_pad {
        Option::Some(auto_pad) => auto_pad,
        Option::None => AUTO_PAD::NOTSET,
    };

    let (out_shape, pads, padded) = match auto_pad {
        AUTO_PAD::NOTSET => {
            let (out_shape, pads) = get_output_shape_explicit_padding(
                pads,
                SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2),
                kernel_shape,
                strides,
                dilations,
                ceil_mode
            );
            let padded = pad_constant_value(X, padding_value, pads);
            (out_shape, pads, padded)
        },
        AUTO_PAD::SAME_UPPER => {
            assert(ceil_mode == 0, 'ceil mode not supp with autopad');
            let out_shape = get_output_shape_auto_pad(
                auto_pad,
                SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2),
                kernel_shape,
                strides
            );
            let pads_shape = get_pad_shape(
                auto_pad,
                SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2),
                kernel_shape,
                strides,
                out_shape
            );

            let pads = get_pad_with_auto_pad(auto_pad, pads_shape);

            let padded = pad_constant_value(X, padding_value, pads);
            (out_shape, pads, padded)
        },
        AUTO_PAD::SAME_LOWER => {
            assert(ceil_mode == 0, 'ceil mode not supp with autopad');
            let out_shape = get_output_shape_auto_pad(
                auto_pad,
                SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2),
                kernel_shape,
                strides
            );
            let pads_shape = get_pad_shape(
                auto_pad,
                SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2),
                kernel_shape,
                strides,
                out_shape
            );

            let pads = get_pad_with_auto_pad(auto_pad, pads_shape);

            let padded = pad_constant_value(X, padding_value, pads);
            (out_shape, pads, padded)
        },
        AUTO_PAD::VALID => {
            assert(ceil_mode == 0, 'ceil mode not supp with autopad');
            let out_shape = get_output_shape_auto_pad(
                auto_pad,
                SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2),
                kernel_shape,
                strides
            );
            let pads_shape = get_pad_shape(
                auto_pad,
                SpanTrait::slice((*X).shape, 2, (*X).shape.len() - 2),
                kernel_shape,
                strides,
                out_shape
            );

            let pads = get_pad_with_auto_pad(auto_pad, pads_shape);

            let padded = pad_constant_value(X, padding_value, pads);
            (out_shape, pads, padded)
        },
    };

    return (
        pool(
            @padded,
            (*X).shape,
            kernel_shape,
            strides,
            out_shape,
            pooling_type,
            pads,
            dilations,
            count_include_pad,
            p,
        ),
        Option::None
    );
}


fn pool<
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
    +Into<i32, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
>(
    padded: @Tensor<T>,
    x_shape: Span<usize>,
    kernel: Span<usize>,
    strides: Option<Span<usize>>,
    out_shape: Span<usize>,
    pooling_type: POOLING_TYPE,
    pads: Span<usize>,
    dilations: Option<Span<usize>>,
    count_include_pad: usize,
    p: usize,
) -> Tensor<T> {
    let n_dims = x_shape.len() - 2;
    let mut y = NullableVecImpl::new();

    let dilations = match dilations {
        Option::Some(dilations) => dilations,
        Option::None => {
            let mut dilations = ArrayTrait::new();
            let mut i = 0;
            loop {
                if i == n_dims {
                    break;
                }
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
            loop {
                if i == n_dims {
                    break;
                }
                strides.append(1);
                i += 1;
            };
            strides.span()
        },
    };
    let mut y_shape = array![*x_shape.at(0), *x_shape.at(1)];
    let mut i = 0;
    loop {
        if i == n_dims {
            break;
        }
        let a: T = NumberTrait::new_unscaled(
            (*x_shape.at(i + 2) + *pads.at(i) + *pads.at(i + n_dims)).into(), false
        );
        let b: T = NumberTrait::new_unscaled(
            ((1 + (*kernel.at(i) - 1) * *dilations.at(i))).into(), false
        );
        let c: T = NumberTrait::new_unscaled((*strides.at(i)).into(), false);
        y_shape.append(NumberTrait::floor(((a - b) / c + NumberTrait::one())).try_into().unwrap());
        i += 1;
    };
    let y_stride = stride(y_shape.span());
    let padded_stride = stride(*padded.shape);
    let mut all_coords = get_all_coords(y_shape.span());

    loop {
        match all_coords.pop_front() {
            Option::Some(coord) => {
                let coord = *coord;
                let window = SpanTrait::slice(
                    *padded.data,
                    *coord.at(0) * *padded_stride.at(0) + *coord.at(1) * *padded_stride.at(1),
                    *padded_stride.at(1)
                );
                let window_stride = SpanTrait::slice(padded_stride, 2, n_dims);
                let mut window_vals = ArrayTrait::new();

                let mut all_indices = ArrayTrait::new();

                let mut i = 0;
                loop {
                    if i == n_dims {
                        break;
                    }
                    let start = *strides.at(i) * *coord.at(i + 2);
                    let end = start + 1 + (*kernel.at(i) - 1) * *dilations.at(i);
                    let step = *dilations.at(i);

                    all_indices.append(arange(start, end, step));

                    i += 1;
                };

                let mut all_indices = cartesian(all_indices.span());

                loop {
                    match all_indices.pop_front() {
                        Option::Some(index) => {
                            let flatten_index = flatten_index((*index), window_stride);

                            window_vals.append(*window.at(flatten_index));
                        },
                        Option::None => { break; }
                    }
                };
                match pooling_type {
                    POOLING_TYPE::AVG => {
                        let flatten_index = flatten_index(coord, y_stride);

                        if count_include_pad == 1 {
                            y.set(flatten_index, average(window_vals.span()));
                        } else {
                            y.set(flatten_index, average(window_vals.span()));
                        }
                    },
                    POOLING_TYPE::LPPOOL => { panic(array!['supported soon']) },
                    POOLING_TYPE::MAX => {
                        let flatten_index = flatten_index(coord, y_stride);

                        y.set(flatten_index, max(window_vals.span()));
                    }
                }
            },
            Option::None => { break; },
        }
    };
    let mut y_data = ArrayTrait::new();
    let mut i = 0;
    loop {
        if i == y.len() {
            break;
        }
        y_data.append(y.at(i));
        i += 1;
    };
    return TensorTrait::new(y_shape.span(), y_data.span());
}

fn get_output_shape_auto_pad(
    auto_pad: AUTO_PAD,
    input_spatial_shape: Span<usize>,
    kernel_spatial_shape: Span<usize>,
    strides_spatial: Option<Span<usize>>,
) -> Span<usize> {
    let n_dims = input_spatial_shape.len();
    let mut out_shape = ArrayTrait::new();

    let strides_spatial = match strides_spatial {
        Option::Some(strides_spatial) => strides_spatial,
        Option::None => {
            let mut strides_spatial = ArrayTrait::new();
            let mut i = 0;
            loop {
                if i == n_dims {
                    break;
                }
                strides_spatial.append(1);
                i += 1;
            };
            strides_spatial.span()
        },
    };

    match auto_pad {
        AUTO_PAD::NOTSET => { panic(array!['not supported!']) },
        AUTO_PAD::SAME_UPPER => {
            let mut i = 0;
            loop {
                if i == input_spatial_shape.len() {
                    break;
                }
                out_shape.append((*input_spatial_shape.at(i) - 1) / *strides_spatial.at(i) + 1);
                i += 1;
            };
            out_shape.span()
        },
        AUTO_PAD::SAME_LOWER => {
            let mut i = 0;
            loop {
                if i == input_spatial_shape.len() {
                    break;
                }
                out_shape.append((*input_spatial_shape.at(i) - 1) / *strides_spatial.at(i) + 1);
                i += 1;
            };
            out_shape.span()
        },
        AUTO_PAD::VALID => {
            let mut i = 0;
            loop {
                if i == input_spatial_shape.len() {
                    break;
                }
                out_shape
                    .append(
                        (*input_spatial_shape.at(i) - *kernel_spatial_shape.at(i))
                            / *strides_spatial.at(i)
                            + 1
                    );
                i += 1;
            };
            out_shape.span()
        },
    }
}

fn get_output_shape_explicit_padding(
    pads: Option<Span<usize>>,
    input_spatial_shape: Span<usize>,
    kernel_spatial_shape: Span<usize>,
    strides_spatial: Option<Span<usize>>,
    dilations: Option<Span<usize>>,
    ceil_mode: usize,
) -> (Span<usize>, Span<usize>) {
    let n_dims = input_spatial_shape.len();
    let pads = match pads {
        Option::Some(pads) => pads,
        Option::None => {
            let mut pads = ArrayTrait::new();
            let mut i = 0;
            loop {
                if i == n_dims {
                    break;
                }
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
            loop {
                if i == n_dims {
                    break;
                }
                dilations.append(1);
                i += 1;
            };
            dilations.span()
        },
    };
    let strides_spatial = match strides_spatial {
        Option::Some(strides_spatial) => strides_spatial,
        Option::None => {
            let mut strides_spatial = ArrayTrait::new();
            let mut i = 0;
            loop {
                if i == n_dims {
                    break;
                }
                strides_spatial.append(1);
                i += 1;
            };
            strides_spatial.span()
        },
    };
    let mut output_spatial_shape = ArrayTrait::<usize>::new();

    let mut d = 0;
    loop {
        if d == n_dims {
            break;
        }
        let dim_num: FP16x16 = NumberTrait::new_unscaled(
            (*input_spatial_shape.at(d)
                + *pads.at(d)
                + *pads.at(d + n_dims)
                - *dilations.at(d) * (*kernel_spatial_shape.at(d) - 1)
                - 1)
                .into(),
            false
        );
        let dim_den = NumberTrait::new_unscaled((*strides_spatial.at(d) + 1).into(), false);

        let dim_size = dim_num / dim_den;

        let oss = if ceil_mode == 1 {
            NumberTrait::ceil(dim_size)
        } else {
            NumberTrait::floor(dim_size)
        };
        output_spatial_shape.append(oss.try_into().unwrap());

        d += 1;
    };
    let output_spatial_shape = output_spatial_shape.span();

    let mut pads_spatial_shape_new_1 = ArrayTrait::new();
    let mut pads_spatial_shape_new_2 = ArrayTrait::new();

    let mut d = 0;
    loop {
        if d == n_dims {
            break;
        }
        let sliding_window_size = (*kernel_spatial_shape.at(d) - 1) * *dilations.at(d) + 1;
        let actual_padded_input_size = (*output_spatial_shape.at(d) - 1) * *strides_spatial.at(d)
            + sliding_window_size;
        let extra_pad_sub = I32Number::new(
            (*input_spatial_shape.at(d) + *pads.at(d) + *pads.at(d + n_dims)).into(), false
        );
        let extra_pad = I32Number::new((actual_padded_input_size).into(), false) - extra_pad_sub;

        if extra_pad > 0 {
            pads_spatial_shape_new_1.append(*pads.at(d) + extra_pad.into() / 2);
            pads_spatial_shape_new_2.append(*pads.at(d) + extra_pad.into() - extra_pad.into() / 2);
        } else {
            pads_spatial_shape_new_1.append(*pads.at(d));
            pads_spatial_shape_new_2.append(*pads.at(d + n_dims));
        };
        d += 1;
    };

    let mut pads_spatial_shape_new = ArrayTrait::new();
    pads_spatial_shape_new.append_span(pads_spatial_shape_new_1.span());
    pads_spatial_shape_new.append_span(pads_spatial_shape_new_2.span());

    return (output_spatial_shape, pads_spatial_shape_new.span());
}


fn get_pad_shape(
    auto_pad: AUTO_PAD,
    input_spatial_shape: Span<usize>,
    kernel_spatial_shape: Span<usize>,
    strides_spatial: Option<Span<usize>>,
    output_spatial_shape: Span<usize>,
) -> Span<usize> {
    let spatial_dims = input_spatial_shape.len();
    let mut pad_shape = ArrayTrait::new();

    let strides_spatial = match strides_spatial {
        Option::Some(strides_spatial) => strides_spatial,
        Option::None => {
            let mut strides_spatial = ArrayTrait::new();
            let mut i = 0;
            loop {
                if i == spatial_dims {
                    break;
                }
                strides_spatial.append(1);
                i += 1;
            };
            strides_spatial.span()
        },
    };

    match auto_pad {
        AUTO_PAD::NOTSET => { panic(array!['not supported!']) },
        AUTO_PAD::SAME_UPPER => {
            let mut i = 0;
            loop {
                if i == spatial_dims {
                    break;
                }
                pad_shape
                    .append(
                        (*output_spatial_shape.at(i) - 1) * *strides_spatial.at(i)
                            + *kernel_spatial_shape.at(i)
                            - *input_spatial_shape.at(i)
                    );
                i += 1;
            };
            pad_shape.span()
        },
        AUTO_PAD::SAME_LOWER => {
            let mut i = 0;
            loop {
                if i == spatial_dims {
                    break;
                }
                pad_shape
                    .append(
                        (*output_spatial_shape.at(i) - 1) * *strides_spatial.at(i)
                            + *kernel_spatial_shape.at(i)
                            - *input_spatial_shape.at(i)
                    );
                i += 1;
            };
            pad_shape.span()
        },
        AUTO_PAD::VALID => {
            let mut i = 0;
            loop {
                if i == input_spatial_shape.len() {
                    break;
                }
                pad_shape.append(0);
                i += 1;
            };
            pad_shape.span()
        },
    }
}


fn get_pad_with_auto_pad(auto_pad: AUTO_PAD, mut pad_shape: Span<usize>,) -> Span<usize> {
    let spatial_dims = pad_shape.len();
    let mut pads = ArrayTrait::new();

    match auto_pad {
        AUTO_PAD::NOTSET => { array![].span() },
        AUTO_PAD::SAME_UPPER => {
            let mut pads_1 = ArrayTrait::new();
            let mut pads_2 = ArrayTrait::new();

            loop {
                match pad_shape.pop_front() {
                    Option::Some(v) => {
                        pads_1.append(*v / 2);
                        pads_2.append(*v - *v / 2);
                    },
                    Option::None => {
                        pads.append_span(pads_1.span());
                        pads.append_span(pads_2.span());
                        break pads.span();
                    }
                }
            }
        },
        AUTO_PAD::SAME_LOWER => {
            let mut pads_1 = ArrayTrait::new();
            let mut pads_2 = ArrayTrait::new();

            loop {
                match pad_shape.pop_front() {
                    Option::Some(v) => {
                        pads_1.append(*v - *v / 2);
                        pads_2.append(*v / 2);
                    },
                    Option::None => {
                        pads.append_span(pads_1.span());
                        pads.append_span(pads_2.span());
                        break pads.span();
                    }
                }
            }
        },
        AUTO_PAD::VALID => {
            let mut i = 0;
            loop {
                if i == spatial_dims {
                    break;
                }
                pads.append(0);
                pads.append(0);
                i += 1;
            };
            pads.span()
        },
    }
}

// X dimension : N x C x d1 x ... x dn, Padding on dimensions d1, ..., dn
fn pad_constant_value<
    T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>, +PrintTrait<T>
>(
    mut X: @Tensor<T>, constant_value: T, pads: Span<usize>
) -> Tensor<T> {
    let n_dims = pads.len() / 2;
    let N = *(*X).shape.at(0);
    let C = *(*X).shape.at(1);

    let mut padded_shape = array![N, C];

    let mut i = 0;
    loop {
        if i == n_dims {
            break;
        }
        padded_shape.append(*(*X).shape.at(i + 2) + *pads.at(i) + *pads.at(i + n_dims));
        i += 1;
    };
    let x_stride = stride((*X).shape);
    let padded_stride = stride(padded_shape.span());

    let window_len = *x_stride.at(1);
    let full_len = *padded_shape.at(0) * *padded_stride.at(0);

    let mut x_padded = full(full_len, constant_value);

    let total_channel = N * C;

    let mut c = 0;
    loop {
        if c == total_channel {
            break;
        }

        let mut i = 0;
        loop {
            if i == window_len {
                break;
            }
            let mut padded_index = c * *padded_stride.at(1);
            let mut flatten_index = i;

            let mut n = 0;
            loop {
                if n == n_dims {
                    break;
                }
                let (ind, rem) = DivRem::div_rem(
                    flatten_index, (*x_stride.at(2 + n)).try_into().unwrap()
                );
                flatten_index = rem;
                padded_index += (ind + *pads.at(n)) * *padded_stride.at(2 + n);
                n += 1;
            };

            x_padded.set(padded_index, *(*X).data.at(c * window_len + i));
            i += 1;
        };
        c += 1;
    };

    let mut padded = ArrayTrait::new();
    let mut i = 0;
    loop {
        if i == x_padded.len() {
            break;
        }
        padded.append(x_padded.at(i));
        i += 1;
    };
    return TensorTrait::new(padded_shape.span(), padded.span());
}


// return a span of len ceil((end - start) / step)
fn full<T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>,>(
    len: usize, fill_value: T
) -> NullableVec<T> {
    let mut full = NullableVecImpl::new();
    let mut i = 0;
    loop {
        if i == len {
            break;
        }
        full.set(i, fill_value);
        i += 1;
    };
    return full;
}


fn flatten_index(index: Span<usize>, stride: Span<usize>) -> usize {
    let mut flatten_index = 0;
    let n = index.len();

    let mut i = 0;
    loop {
        if i == n {
            break;
        }
        flatten_index += *index.at(i) * *stride.at(i);
        i += 1;
    };

    return flatten_index;
}


fn get_all_coords(shape: Span<usize>) -> Span<Span<usize>> {
    let mut all_indices = ArrayTrait::new();

    let mut i = 0;
    loop {
        if i == shape.len() {
            break;
        }
        all_indices.append(arange(0, *shape.at(i), 1));
        i += 1;
    };

    return cartesian(all_indices.span());
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
    let mut size_arrays = ArrayTrait::new();
    loop {
        if i == arrays.len() {
            break;
        }
        size_arrays.append((*(arrays.at(i))).len());

        i += 1;
    };
    let size_arrays = size_arrays.span();
    let mut output_arrays = ArrayTrait::<Array<usize>>::new();
    let mut m = n;

    let mut i = 0;
    loop {
        if i == arrays.len() {
            break;
        }
        m = m / (*(arrays.at(i))).len();
        let mut out = repeat(*(arrays.at(i)), m);
        out = repeat_2(out, size_arrays, i);

        output_arrays.append(out);
        i += 1;
    };
    let output_arrays = output_arrays.span();

    let mut i = 0;
    let mut ret = ArrayTrait::new();
    loop {
        if i == n {
            break;
        }
        let mut j = 0;
        let mut x = ArrayTrait::new();
        loop {
            if j == arrays.len() {
                break;
            }

            x.append(*(output_arrays.at(j)).at(i));
            j += 1;
        };
        ret.append(x.span());
        i += 1;
    };

    return ret.span();
}


fn repeat_2(mut array: Array<usize>, size_array: Span<usize>, index: usize) -> Array<usize> {
    let mut size = array.len();
    let mut i = 0;
    loop {
        if i == index {
            break;
        }
        let mut j = 1;
        loop {
            if j == *size_array.at(index - 1 - i) {
                break;
            }
            let mut k = 0;
            loop {
                if k == size {
                    break;
                }
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
    let mut out = ArrayTrait::new();
    let mut j = 0;
    loop {
        if j == array.len() {
            break;
        }
        let mut k = 0;
        loop {
            if k == m {
                break;
            }
            out.append(*array.at(j));
            k += 1;
        };
        j += 1;
    };

    out
}


fn arange(start: usize, end: usize, step: usize) -> Span<usize> {
    let mut arr = ArrayTrait::new();
    let mut i = start;
    loop {
        if i >= end {
            break;
        }
        arr.append(i);
        i += step;
    };
    return arr.span();
}


fn max<T, MAG, +NumberTrait<T, MAG>, +Drop<T>, +Copy<T>, +PartialOrd<T>,>(mut a: Span<T>) -> T {
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


fn average<
    T,
    MAG,
    +NumberTrait<T, MAG>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Drop<T>,
    +Copy<T>,
    +PartialOrd<T>,
    +Div<T>
>(
    mut a: Span<T>
) -> T {
    assert(a.len() > 0, 'span cannot be empty');

    let mut sum = *a.at(0);
    let n = NumberTrait::new_unscaled((a.len()).into(), false);
    loop {
        match a.pop_front() {
            Option::Some(v) => { sum += *v; },
            Option::None => { break sum / n; }
        };
    }
}

