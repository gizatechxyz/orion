use alexandria_sorting::BubbleSort;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{
    TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, BoolTensor
};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};

#[derive(Copy, Drop)]
enum MODE {
    NEAREST,
    LINEAR,
    CUBIC,
}

#[derive(Copy, Drop)]
enum NEAREST_MODE {
    ROUND_PREFER_FLOOR,
    ROUND_PREFER_CEIL,
    FLOOR,
    CEIL
}

#[derive(Copy, Drop)]
enum KEEP_ASPECT_RATIO_POLICY {
    STRETCH,
    NOT_LARGER,
    NOT_SMALLER
}

#[derive(Copy, Drop)]
enum TRANSFORMATION_MODE {
    HALF_PIXEL,
    ALIGN_CORNERS,
    ASYMMETRIC,
    TF_CROP_AND_RESIZE,
    PYTORCH_HALF_PIXEL,
    HALF_PIXEL_SYMMETRIC
}

/// Cf: TensorTrait::resize docstring
fn resize<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    self: @Tensor<T>,
    roi: Option<Tensor<T>>,
    scales: Option<Span<T>>,
    sizes: Option<Span<usize>>,
    antialias: Option<usize>,
    axes: Option<Span<usize>>,
    coordinate_transformation_mode: Option<TRANSFORMATION_MODE>,
    cubic_coeff_a: Option<T>,
    exclude_outside: Option<bool>,
    extrapolation_value: Option<T>,
    keep_aspect_ratio_policy: Option<KEEP_ASPECT_RATIO_POLICY>,
    mode: Option<MODE>,
    nearest_mode: Option<NEAREST_MODE>,
) -> Tensor<T> {
    let output = interpolate_nd(
        self,
        antialias,
        mode,
        nearest_mode,
        scales,
        sizes,
        roi,
        keep_aspect_ratio_policy,
        exclude_outside,
        coordinate_transformation_mode,
        extrapolation_value,
        axes,
        cubic_coeff_a
    );

    output
}

fn interpolate_nd<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    data: @Tensor<T>,
    antialias: Option<usize>,
    mode: Option<MODE>,
    nearest_mode: Option<NEAREST_MODE>,
    scale_factors: Option<Span<T>>,
    output_size: Option<Span<usize>>,
    roi: Option<Tensor<T>>,
    keep_aspect_ratio_policy: Option<KEEP_ASPECT_RATIO_POLICY>,
    exclude_outside: Option<bool>,
    coordinate_transformation_mode: Option<TRANSFORMATION_MODE>,
    extrapolation_value: Option<T>,
    axes: Option<Span<usize>>,
    cubic_coeff_a: Option<T>,
) -> Tensor<T> {
    let mode = match mode {
        Option::Some(mode) => mode,
        Option::None => { MODE::NEAREST },
    };

    let keep_aspect_ratio_policy = match keep_aspect_ratio_policy {
        Option::Some(keep_aspect_ratio_policy) => keep_aspect_ratio_policy,
        Option::None => { KEEP_ASPECT_RATIO_POLICY::STRETCH },
    };

    let exclude_outside = match exclude_outside {
        Option::Some(exclude_outside) => exclude_outside,
        Option::None => { false },
    };

    let extrapolation_value = match extrapolation_value {
        Option::Some(extrapolation_value) => extrapolation_value,
        Option::None => { NumberTrait::zero() },
    };

    if output_size.is_none() && scale_factors.is_none() {
        core::panic_with_felt252('size and scale are None');
    }

    let r = (*data).shape.len();

    let (axes, scale_factors, output_size, roi) = match axes {
        Option::Some(axes) => {
            let mut scale_factors = match scale_factors {
                Option::Some(scale_factors) => {
                    let mut new_scale_factors = ArrayTrait::<T>::new();
                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break NumberTrait::one();
                            }

                            if *axes.at(i) == d {
                                break *scale_factors.at(i);
                            }

                            i += 1;
                        };
                        new_scale_factors.append(item);
                        d += 1;
                    };

                    Option::Some(new_scale_factors.span())
                },
                Option::None => { Option::None },
            };

            let mut output_size = match output_size {
                Option::Some(output_size) => {
                    let mut new_output_size = array![];
                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break *(*data).shape.at(d);
                            }

                            if *axes.at(i) == d {
                                break *output_size.at(i);
                            }

                            i += 1;
                        };
                        new_output_size.append(item);
                        d += 1;
                    };

                    Option::Some(new_output_size.span())
                },
                Option::None => { Option::None },
            };

            let mut roi = match roi {
                Option::Some(roi) => {
                    let mut new_roi_data = array![];
                    let naxes = axes.len();
                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break NumberTrait::zero();
                            }

                            if *axes.at(i) == d {
                                break *roi.data.at(i);
                            }

                            i += 1;
                        };

                        new_roi_data.append(item);
                        d += 1;
                    };

                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break NumberTrait::one();
                            }

                            if *axes.at(i) == d {
                                break *roi.data.at(i + naxes);
                            }

                            i += 1;
                        };

                        new_roi_data.append(item);
                        d += 1;
                    };

                    let mut shape = ArrayTrait::new();
                    shape.append(r * 2);
                    Option::Some(TensorTrait::new(shape.span(), new_roi_data.span()))
                },
                Option::None => { Option::None },
            };

            (axes, scale_factors, output_size, roi)
        },
        Option::None => {
            let mut axes = array![];
            let mut i = 0;
            while i != r {
                axes.append(i);
                i += 1;
            };

            (axes.span(), scale_factors, output_size, roi)
        }
    };
    let (mut output_size, mut scale_factors) = match output_size {
        Option::Some(output_size) => {
            let mut scale_factors: Array<T> = array![];
            let mut i = 0;
            while i != r {
                let output_size_i: T = NumberTrait::new_unscaled(
                    (*output_size.at(i)).into(), false
                );
                let data_shape_i: T = NumberTrait::new_unscaled(
                    (*(*data).shape.at(i)).into(), false
                );

                scale_factors.append(output_size_i / data_shape_i);
                i += 1;
            };

            let (mut output_size, mut scale_factors) = match keep_aspect_ratio_policy {
                KEEP_ASPECT_RATIO_POLICY::STRETCH => { (output_size, scale_factors.span()) },
                KEEP_ASPECT_RATIO_POLICY::NOT_LARGER => {
                    let mut scale = *scale_factors.at(*axes.at(0));
                    let mut i = 1;
                    while i != axes
                        .len() {
                            if scale > *scale_factors.at(*axes.at(i)) {
                                scale = *scale_factors.at(*axes.at(i));
                            }

                            i += 1;
                        };

                    let mut scale_factors: Array<T> = array![];
                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break NumberTrait::one();
                            }

                            if *axes.at(i) == d {
                                break scale;
                            }

                            i += 1;
                        };
                        scale_factors.append(item);
                        d += 1;
                    };

                    let mut output_size = array![];
                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break *(*data).shape.at(d);
                            }

                            if *axes.at(i) == d {
                                break NumberTrait::round(
                                    scale
                                        * NumberTrait::new_unscaled(
                                            (*(*data).shape.at(d)).into(), false
                                        )
                                )
                                    .try_into()
                                    .unwrap();
                            }

                            i += 1;
                        };
                        output_size.append(item);
                        d += 1;
                    };

                    (output_size.span(), scale_factors.span())
                },
                KEEP_ASPECT_RATIO_POLICY::NOT_SMALLER => {
                    let mut scale = *scale_factors.at(*axes.at(0));
                    let mut i = 1;
                    while i != axes
                        .len() {
                            if scale < *scale_factors.at(*axes.at(i)) {
                                scale = *scale_factors.at(*axes.at(i));
                            }

                            i += 1;
                        };

                    let mut scale_factors: Array<T> = array![];
                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break NumberTrait::one();
                            }

                            if *axes.at(i) == d {
                                break scale;
                            }

                            i += 1;
                        };
                        scale_factors.append(item);
                        d += 1;
                    };

                    let mut output_size = array![];
                    let mut d = 0;
                    while d != r {
                        let mut i = 0;
                        let item = loop {
                            if i == axes.len() {
                                break *(*data).shape.at(d);
                            }

                            if *axes.at(i) == d {
                                break NumberTrait::round(
                                    scale
                                        * NumberTrait::new_unscaled(
                                            (*(*data).shape.at(d)).into(), false
                                        )
                                )
                                    .try_into()
                                    .unwrap();
                            }

                            i += 1;
                        };
                        output_size.append(item);
                        d += 1;
                    };

                    (output_size.span(), scale_factors.span())
                },
            };

            (output_size, scale_factors)
        },
        Option::None => {
            let mut output_size: Array<usize> = array![];

            let scale_factors = match scale_factors {
                Option::Some(scale_factors) => scale_factors,
                Option::None => { core::panic_with_felt252('size and scale None') },
            };

            let mut i = 0;
            while i != scale_factors
                .len() {
                    let item = *scale_factors.at(i)
                        * NumberTrait::new_unscaled((*(*(data).shape).at(i)).into(), false);
                    output_size.append(item.try_into().unwrap());
                    i += 1;
                };

            (output_size.span(), scale_factors)
        },
    };

    let mut ret: Array<Span<usize>> = array![];
    let mut i = 0;
    while i != output_size
        .len() {
            let mut temp = ArrayTrait::<usize>::new();
            let mut j = 0;
            while j != *output_size.at(i) {
                temp.append(j);
                j += 1;
            };

            ret.append(temp.span());
            i += 1;
        };

    let mut ret = cartesian(ret.span());
    let mut ret_data = array![];

    loop {
        match ret.pop_front() {
            Option::Some(X) => {
                let mut x: Array<T> = array![];
                let mut i = 0;
                while i != X
                    .len() {
                        x.append(NumberTrait::new_unscaled((*X.at(i)).into(), false));
                        i += 1;
                    };

                let mut x = x.span();
                let item = interpolate_nd_with_x(
                    data,
                    (*data).shape.len(),
                    scale_factors,
                    output_size,
                    x,
                    antialias,
                    mode,
                    nearest_mode,
                    roi,
                    extrapolation_value,
                    coordinate_transformation_mode,
                    exclude_outside,
                    cubic_coeff_a
                );

                ret_data.append(*item.data.at(0));
            },
            Option::None => { break; }
        }
    };

    let mut shape = array![];
    shape.append(ret_data.len());

    TensorTrait::new(output_size, ret_data.span())
}

fn cartesian(mut arrays: Span<Span<usize>>,) -> Array<Array<usize>> {
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
    let mut size_arrays = array![];
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
    let mut ret = array![];
    while i != n {
        let mut j = 0;
        let mut x = array![];
        while j != arrays.len() {
            x.append(*(output_arrays.at(j)).at(i));
            j += 1;
        };

        ret.append(x);
        i += 1;
    };

    ret
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
    let mut out = array![];
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

fn interpolate_nd_with_x<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    data: @Tensor<T>,
    n: usize,
    mut scale_factor: Span<T>,
    mut output_size: Span<usize>,
    mut x: Span<T>,
    antialias: Option<usize>,
    mode: MODE,
    nearest_mode: Option<NEAREST_MODE>,
    roi: Option<Tensor<T>>,
    extrapolation_value: T,
    coordinate_transformation_mode: Option<TRANSFORMATION_MODE>,
    exclude_outside: bool,
    cubic_coeff_a: Option<T>,
) -> Tensor<T> {
    if n == 1 {
        return interpolate_1d_with_x(
            data,
            *scale_factor.at(0),
            *output_size.at(0),
            *x.at(0),
            antialias,
            mode,
            nearest_mode,
            roi,
            extrapolation_value,
            coordinate_transformation_mode,
            exclude_outside,
            cubic_coeff_a
        );
    }

    let mut res1d = array![];

    let scale_factor_zero = match scale_factor.pop_front() {
        Option::Some(item) => { *item },
        Option::None => core::panic_with_felt252('scale factor empty')
    };
    let output_size_zero = match output_size.pop_front() {
        Option::Some(item) => { *item },
        Option::None => core::panic_with_felt252('output_size empty')
    };
    let x_zero = match x.pop_front() {
        Option::Some(item) => { *item },
        Option::None => core::panic_with_felt252('x empty')
    };

    let reduced_roi = match roi {
        Option::Some(roi) => {
            let mut reduced_roi = ArrayTrait::new();
            let mut reduced_roi_shape = ArrayTrait::new();
            reduced_roi_shape.append(roi.data.len() - 2);

            let mut i = 1;
            while i != 2 * n {
                if i != n {
                    reduced_roi.append(*roi.data.at(i));
                }

                i += 1;
            };
            Option::Some(TensorTrait::new(reduced_roi_shape.span(), reduced_roi.span()))
        },
        Option::None => { Option::None }
    };

    let mut i = 0;
    while i != *(*data)
        .shape
        .at(0) {
            let data = get_row_n(data, i);

            let mut r = interpolate_nd_with_x(
                @data,
                n - 1,
                scale_factor,
                output_size,
                x,
                antialias,
                mode,
                nearest_mode,
                reduced_roi,
                extrapolation_value,
                coordinate_transformation_mode,
                exclude_outside,
                cubic_coeff_a
            );

            loop {
                match r.data.pop_front() {
                    Option::Some(item) => { res1d.append(*item); },
                    Option::None => { break; }
                }
            };

            i += 1;
        };

    let mut shape = array![];
    shape.append(res1d.len());

    let res1d = TensorTrait::new(shape.span(), res1d.span());

    let reduced_roi = match roi {
        Option::Some(roi) => {
            let mut reduced_roi = array![];
            let mut reduced_roi_shape = array![];

            reduced_roi_shape.append(2);
            reduced_roi.append(*roi.data.at(0));
            reduced_roi.append(*roi.data.at(n));

            Option::Some(TensorTrait::new(reduced_roi_shape.span(), reduced_roi.span()))
        },
        Option::None => { Option::None }
    };

    let a = interpolate_1d_with_x(
        @res1d,
        scale_factor_zero,
        output_size_zero,
        x_zero,
        antialias,
        mode,
        nearest_mode,
        reduced_roi,
        extrapolation_value,
        coordinate_transformation_mode,
        exclude_outside,
        cubic_coeff_a
    );

    //let mut ret = array![];
    //let mut shape = array![];
    //shape.append(2);
    //ret.append(NumberTrait::zero());

    a
}

fn get_row_n<T, +TensorTrait<T>, +Copy<T>, +Drop<T>,>(
    data: @Tensor<T>, index: usize,
) -> Tensor<T> {
    let mut output_data = array![];
    let mut output_shape = array![];
    let mut stride_output = 1;

    let mut i = 0;
    while i != (*data)
        .shape
        .len() {
            if i != 0 {
                output_shape.append(*(*data).shape.at(i));
                stride_output = stride_output * *(*data).shape.at(i);
            }

            i += 1;
        };

    let mut i = 0;
    while i != stride_output {
        output_data.append(*(*data).data.at(index * stride_output + i));
        i += 1;
    };

    TensorTrait::new(output_shape.span(), output_data.span())
}

fn interpolate_1d_with_x<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    data: @Tensor<T>,
    scale_factor: T,
    output_width_int: usize,
    x: T,
    antialias: Option<usize>,
    mode: MODE,
    nearest_mode: Option<NEAREST_MODE>,
    roi: Option<Tensor<T>>,
    extrapolation_value: T,
    coordinate_transformation_mode: Option<TRANSFORMATION_MODE>,
    exclude_outside: bool,
    cubic_coeff_a: Option<T>,
) -> Tensor<T> {
    let coordinate_transformation_mode = match coordinate_transformation_mode {
        Option::Some(coordinate_transformation_mode) => coordinate_transformation_mode,
        Option::None => { TRANSFORMATION_MODE::HALF_PIXEL },
    };

    let input_width = (*data).data.len();
    let output_width = (scale_factor * NumberTrait::new_unscaled((input_width).into(), false));

    let x_ori: T = match coordinate_transformation_mode {
        TRANSFORMATION_MODE::HALF_PIXEL => {
            (x + NumberTrait::half()) / scale_factor - NumberTrait::half()
        },
        TRANSFORMATION_MODE::ALIGN_CORNERS => {
            let mut x_ori = NumberTrait::zero();
            if output_width != NumberTrait::one() {
                x_ori = x
                    * (NumberTrait::new_unscaled(input_width.into(), false) - NumberTrait::one())
                    / (output_width - NumberTrait::one());
            }
            x_ori
        },
        TRANSFORMATION_MODE::ASYMMETRIC => { x / scale_factor },
        TRANSFORMATION_MODE::TF_CROP_AND_RESIZE => {
            let x_ori = match roi {
                Option::Some(roi) => {
                    let mut x_ori = if output_width == NumberTrait::one() {
                        (*roi.data.at(1) - *roi.data.at(0))
                            * (NumberTrait::new_unscaled(input_width.into(), false)
                                - NumberTrait::one())
                            / (NumberTrait::one() + NumberTrait::one())
                    } else {
                        x
                            * (*roi.data.at(1) - *roi.data.at(0))
                            * (NumberTrait::new_unscaled(input_width.into(), false)
                                - NumberTrait::one())
                            / (output_width - NumberTrait::one())
                    };

                    x_ori = x_ori
                        + *roi.data.at(0)
                            * (NumberTrait::new_unscaled(input_width.into(), false)
                                - NumberTrait::one());

                    if x_ori < NumberTrait::zero()
                        || x_ori > (NumberTrait::new_unscaled(input_width.into(), false)
                            - NumberTrait::one()) {
                        let mut ret = ArrayTrait::new();
                        let mut shape = ArrayTrait::new();
                        shape.append(1);
                        ret.append(extrapolation_value);
                        return TensorTrait::new(shape.span(), ret.span());
                    };
                    x_ori
                },
                Option::None => { core::panic_with_felt252('roi cannot be None.') },
            };
            x_ori
        },
        TRANSFORMATION_MODE::PYTORCH_HALF_PIXEL => {
            if output_width == NumberTrait::one() {
                NumberTrait::neg(NumberTrait::<T>::half())
            } else {
                (x + NumberTrait::half()) / scale_factor - NumberTrait::half()
            }
        },
        TRANSFORMATION_MODE::HALF_PIXEL_SYMMETRIC => {
            let adjustement: T = NumberTrait::new_unscaled(output_width_int.into(), false)
                / output_width;
            let center: T = NumberTrait::new_unscaled(input_width.into(), false)
                / (NumberTrait::one() + NumberTrait::one());
            let offset = center * (NumberTrait::one() - adjustement);
            offset + (x + NumberTrait::half()) / scale_factor - NumberTrait::half()
        },
    };

    let x_ori_int = x_ori.floor();

    let ratio = if x_ori_int.try_into().unwrap() == x_ori {
        NumberTrait::one()
    } else {
        x_ori - x_ori_int.try_into().unwrap()
    };

    let mut coeffs = match mode {
        MODE::NEAREST => {
            let coeffs = match antialias {
                Option::Some => core::panic_with_felt252('antialias not for mode NEAREST'),
                Option::None => { nearest_coeffs(ratio, nearest_mode) },
            };
            coeffs
        },
        MODE::LINEAR => {
            let coeffs = match antialias {
                Option::Some(antialias) => {
                    let coeffs = if antialias == 0 {
                        linear_coeffs(ratio)
                    } else {
                        linear_coeffs_antialias(ratio, scale_factor)
                    };
                    coeffs
                },
                Option::None => { linear_coeffs(ratio) },
            };
            coeffs
        },
        MODE::CUBIC => {
            let coeffs = match antialias {
                Option::Some => { cubic_coeffs_antialias(ratio, scale_factor, cubic_coeff_a) },
                Option::None => { cubic_coeffs(ratio, cubic_coeff_a) },
            };
            coeffs
        },
    };

    let n = coeffs.data.len();

    let (idxes, points) = get_neighbor(x_ori, n, data);

    if exclude_outside {
        let mut coeffs_exclude_outside: Array<T> = array![];
        let mut sum = NumberTrait::zero();
        let mut i = 0;
        while i != idxes
            .data
            .len() {
                if *idxes.data.at(i) {
                    coeffs_exclude_outside.append(NumberTrait::zero());
                    sum += NumberTrait::zero();
                } else {
                    coeffs_exclude_outside.append(*coeffs.data.at(i));
                    sum += *coeffs.data.at(i);
                }

                i += 1;
            };

        let mut coeff_div: Array<T> = array![];
        let mut i = 0;
        while i != n {
            coeff_div.append(*coeffs_exclude_outside.at(i) / sum);
            i += 1;
        };

        coeffs = TensorTrait::new(coeffs.shape, coeff_div.span());
    }

    TensorTrait::matmul(@coeffs, @points)
}

fn get_neighbor<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    mut x: T, n: usize, data: @Tensor<T>,
) -> (Tensor<bool>, Tensor<T>) {
    let pad_width: usize = NumberTrait::ceil(
        NumberTrait::new_unscaled(n.into(), false)
            / (NumberTrait::<T>::one() + NumberTrait::<T>::one())
    )
        .try_into()
        .unwrap();
    let mut padded = array![];

    let mut i = 0;
    while i != pad_width {
        padded.append(*(*data).data.at(0));
        i += 1;
    };

    let mut i = 0;
    while i != (*data).data.len() {
        padded.append(*(*data).data.at(i));
        i += 1;
    };

    let mut i = 0;
    while i != pad_width {
        padded.append(*(*data).data.at((*data).data.len() - 1));
        i += 1;
    };

    x = x + NumberTrait::new_unscaled(pad_width.into(), false);

    let mut idxes = get_neighbor_idxes(x, n, padded.len());

    let mut idxes_centered = array![];
    let mut ret = array![];
    let mut i = 0;
    while i != idxes
        .data
        .len() {
            ret.append(*padded.at(*idxes.data.at(i)));

            if *idxes.data.at(i) >= pad_width {
                if (*idxes.data.at(i) - pad_width) >= (*data).data.len() {
                    idxes_centered.append(true);
                } else {
                    idxes_centered.append(false);
                }
            } else {
                idxes_centered.append(true);
            }

            i += 1;
        };

    let mut shape = array![];
    shape.append(idxes.data.len());

    (
        TensorTrait::new(shape.span(), idxes_centered.span()),
        TensorTrait::new(shape.span(), ret.span())
    )
}

fn get_neighbor_idxes<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    mut x: T, n: usize, limit: usize,
) -> Tensor<usize> {
    let _pad_width: usize = NumberTrait::<
        T
    >::ceil(
        NumberTrait::new_unscaled(n.into(), false)
            / (NumberTrait::<T>::one() + NumberTrait::<T>::one())
    )
        .try_into()
        .unwrap();
    let mut idxes = array![];

    if n % 2 == 0 {
        let (mut i_low, mut i_high) = if x < NumberTrait::zero() {
            (0, 1)
        } else {
            (NumberTrait::floor(x).try_into().unwrap(), NumberTrait::ceil(x).try_into().unwrap())
        };

        if i_high >= limit {
            i_low = limit - 2;
            i_high = limit - 1;
        }

        if i_low == i_high {
            if i_low == 0 {
                i_high = i_high + 1;
            } else {
                i_low = i_low - 1;
            }
        }

        let mut i = 0;
        while i != n
            / 2 {
                if i_low - i < 0 {
                    idxes.append(i_high + i);
                    i_high += 1;
                } else {
                    idxes.append(i_low - i);
                }
                if i_high + i >= limit {
                    i_low -= 1;
                    idxes.append(i_low - i);
                } else {
                    idxes.append(i_high + i);
                }

                i += 1;
            }
    } else {
        core::panic_with_felt252('MUST BE EVEN');
    }

    idxes = BubbleSort::sort(idxes.span());

    let mut shape = array![];
    shape.append(n);

    TensorTrait::new(shape.span(), idxes.span())
}

fn linear_coeffs<
    T,
    MAG,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TensorTrait<T>,
    +Copy<T>,
    +Drop<T>,
    +Sub<T>
>(
    mut ratio: T
) -> Tensor<T> {
    let mut ret = array![];
    let mut shape = array![];
    shape.append(2);
    ret.append(NumberTrait::one() - ratio);
    ret.append(ratio);

    TensorTrait::new(shape.span(), ret.span())
}

fn linear_coeffs_antialias<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    mut ratio: T, scale: T
) -> Tensor<T> {
    let scale = NumberTrait::min(scale, NumberTrait::one());
    let start = (NumberTrait::floor(NumberTrait::neg(NumberTrait::one()) / scale)
        + NumberTrait::one());
    let footprint = (NumberTrait::one() + NumberTrait::one())
        - (NumberTrait::one() + NumberTrait::one()) * start;

    let mut coeffs: Array<T> = array![];
    let mut sum = NumberTrait::zero();

    // arange and clip + compute sum
    let mut i = start;
    while i != start
        + footprint {
            let value = NumberTrait::one() - NumberTrait::abs((i - ratio) * scale);

            if value < NumberTrait::zero() {
                coeffs.append(NumberTrait::zero());
            } else if value > NumberTrait::one() {
                coeffs.append(NumberTrait::one());
                sum += NumberTrait::one();
            } else {
                coeffs.append(value);
                sum += value;
            }

            i += NumberTrait::one();
        };

    let n = coeffs.len();

    let mut coeff_div: Array<T> = array![];
    let mut i = 0;
    while i != n {
        coeff_div.append(*coeffs.at(i) / sum);
        i += 1;
    };

    let mut shape = array![];
    shape.append(n);

    TensorTrait::new(shape.span(), coeff_div.span())
}

fn cubic_coeffs<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    mut ratio: T, A: Option<T>
) -> Tensor<T> {
    let one = NumberTrait::one();
    let two = one + NumberTrait::one();
    let three = two + NumberTrait::one();
    let four = three + NumberTrait::one();
    let five = four + NumberTrait::one();
    let eigth = four + four;

    let A = match A {
        Option::Some(A) => A,
        Option::None => { NumberTrait::neg(three / four) },
    };

    let mut coeffs = array![];
    let mut shape = array![];

    coeffs
        .append(
            ((A * (ratio + one) - five * A) * (ratio + one) + eigth * A) * (ratio + one) - four * A
        );
    coeffs.append(((A + two) * ratio - (A + three)) * ratio * ratio + one);
    coeffs.append(((A + two) * (one - ratio) - (A + three)) * (one - ratio) * (one - ratio) + one);
    coeffs
        .append(
            ((A * ((one - ratio) + one) - five * A) * ((one - ratio) + one) + eigth * A)
                * ((one - ratio) + one)
                - four * A
        );

    shape.append(4);

    TensorTrait::new(shape.span(), coeffs.span())
}

fn cubic_coeffs_antialias<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    mut ratio: T, scale: T, A: Option<T>
) -> Tensor<T> {
    let one = NumberTrait::one();
    let two = one + NumberTrait::one();
    let three = two + NumberTrait::one();
    let four = three + NumberTrait::one();

    let scale = NumberTrait::min(scale, NumberTrait::one());

    let i_start = NumberTrait::floor(NumberTrait::neg(two) / scale) + NumberTrait::one();
    let i_end = two - i_start;
    assert(i_end > i_start, 'i_end must be greater');

    let A = match A {
        Option::Some(A) => A,
        Option::None => { NumberTrait::neg(three / four) },
    };

    let mut coeffs = array![];
    let mut sum = NumberTrait::zero();

    let mut i = i_start;
    while i != i_end {
        let value = compute_coeff(scale * (i - ratio), A);
        coeffs.append(value);
        sum += value;

        i += NumberTrait::one();
    };

    let n = coeffs.len();

    let mut coeff_div: Array<T> = array![];
    let mut i = 0;
    while i != n {
        coeff_div.append(*coeffs.at(i) / sum);
        i += 1;
    };

    let mut shape = array![];
    shape.append(n);

    TensorTrait::new(shape.span(), coeff_div.span())
}

fn compute_coeff<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    mut x: T, A: T
) -> T {
    let one = NumberTrait::one();
    let two = one + NumberTrait::one();
    let three = two + NumberTrait::one();
    let four = three + NumberTrait::one();
    let five = four + NumberTrait::one();
    let eigth = four + four;

    x = x.abs();
    let mut x_2 = x * x;
    let mut x_3 = x * x_2;
    if x <= one {
        return (A + two) * x_3 - (A + three) * x_2 + one;
    }
    if x < two {
        return A * x_3 - five * A * x_2 + eigth * A * x - four * A;
    }

    NumberTrait::zero()
}

fn nearest_coeffs<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +AddEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Sub<T>,
>(
    mut ratio: T, nearest_mode: Option<NEAREST_MODE>
) -> Tensor<T> {
    let nearest_mode = match nearest_mode {
        Option::Some(nearest_mode) => { nearest_mode },
        Option::None => { NEAREST_MODE::ROUND_PREFER_FLOOR },
    };

    let mut ret = array![];
    let mut shape = array![];
    shape.append(2);

    // CHECK SI C'EST UNE CONDITION ASSEZ GENERALE
    if ratio == NumberTrait::one() {
        ret.append(NumberTrait::zero());
        ret.append(NumberTrait::one());
        return TensorTrait::new(shape.span(), ret.span());
    }

    match nearest_mode {
        NEAREST_MODE::ROUND_PREFER_FLOOR => {
            if ratio <= NumberTrait::half() {
                ret.append(NumberTrait::one());
                ret.append(NumberTrait::zero());
                return TensorTrait::new(shape.span(), ret.span());
            } else {
                ret.append(NumberTrait::zero());
                ret.append(NumberTrait::one());
                return TensorTrait::new(shape.span(), ret.span());
            }
        },
        NEAREST_MODE::ROUND_PREFER_CEIL => {
            if ratio < NumberTrait::half() {
                ret.append(NumberTrait::one());
                ret.append(NumberTrait::zero());
                return TensorTrait::new(shape.span(), ret.span());
            } else {
                ret.append(NumberTrait::zero());
                ret.append(NumberTrait::one());
                return TensorTrait::new(shape.span(), ret.span());
            }
        },
        NEAREST_MODE::FLOOR => {
            ret.append(NumberTrait::one());
            ret.append(NumberTrait::zero());
            return TensorTrait::new(shape.span(), ret.span());
        },
        NEAREST_MODE::CEIL => {
            ret.append(NumberTrait::zero());
            ret.append(NumberTrait::one());
            return TensorTrait::new(shape.span(), ret.span());
        },
    }
}

