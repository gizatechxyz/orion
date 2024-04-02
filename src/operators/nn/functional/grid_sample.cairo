use core::debug::PrintTrait;

use orion::numbers::FP16x16;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{stride};
use orion::operators::tensor::{FP16x16Tensor, TensorTrait, Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};

#[derive(Copy, Drop)]
enum MODE {
    NEAREST,
    LINEAR,
    CUBIC,
}

#[derive(Copy, Drop)]
enum PADDING_MODE {
    ZEROS,
    BORDER,
    REFLECTION,
}

fn grid_sample<
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
    grid: @Tensor<T>,
    align_corner: Option<usize>,
    mode: Option<MODE>,
    padding_mode: Option<PADDING_MODE>,
) -> Tensor<T> {
    let align_corner = match align_corner {
        Option::Some(align_corner) => align_corner,
        Option::None => 0,
    };

    let mode = match mode {
        Option::Some(mode) => mode,
        Option::None => MODE::LINEAR,
    };

    let padding_mode = match padding_mode {
        Option::Some(padding_mode) => padding_mode,
        Option::None => PADDING_MODE::ZEROS,
    };

    let x_dims = (*X).shape;
    let x_stride = stride((*X).shape);
    let grid_dims = (*grid).shape;
    let grid_stride = stride((*grid).shape);

    let N = *x_dims.at(0);
    let C = *x_dims.at(1);

    let num_dims = x_dims.len() - 2;
    let dims = SpanTrait::slice(x_dims, 2, num_dims);

    let border = prepare_border(X, dims, align_corner);

    let mut y_dims: Array<usize> = array![N, C];
    y_dims.append_span(SpanTrait::slice(grid_dims, 1, grid_dims.len() - 2));
    let y_dims = y_dims.span();

    if prod(y_dims, 0) == 0 {
        return TensorTrait::new(array![].span(), array![].span());
    }

    let mut Y: Array<T> = array![];

    let mut n = 0;
    while n != N {
        let grid_data = SpanTrait::slice((*grid).data, n * *grid_stride.at(0), *grid_stride.at(0));
        let grid_data_stride = SpanTrait::slice(grid_stride, 1, grid_stride.len() - 1);

        let mut c = 0;
        while c != C {
            let X_data = SpanTrait::slice(
                (*X).data, n * *x_stride.at(0) + c * *x_stride.at(1), *x_stride.at(1)
            );
            let X_data_stride = SpanTrait::slice(x_stride, 2, grid_stride.len() - 2);
            let all_coords = get_all_coords(SpanTrait::slice(grid_dims, 1, grid_dims.len() - 2));

            let mut ix = 0;
            while ix != all_coords
                .len() {
                    let ox = *all_coords.at(ix);
                    let nx = get_sub(grid_data, grid_data_stride, ox);
                    let nx = reverse(nx);
                    let x = gs_denormalize_coordinates(nx, dims, align_corner);

                    let x = match mode {
                        MODE::NEAREST => { rint(x) },
                        MODE::LINEAR => { x },
                        MODE::CUBIC => { x },
                    };

                    let mut new_x: Array<T> = array![];
                    let mut i = 0;
                    while i != x
                        .len() {
                            let v = *x.at(i);
                            let mut x_min = *border.at(i);
                            let mut x_max = *border.at(i + num_dims);
                            let new_v = if v < x_min || v > x_max {
                                let v = match padding_mode {
                                    PADDING_MODE::ZEROS => { v },
                                    PADDING_MODE::BORDER => {
                                        clamp(
                                            v,
                                            NumberTrait::zero(),
                                            NumberTrait::new_unscaled((*dims.at(i)).into(), false)
                                                - NumberTrait::one()
                                        )
                                    },
                                    PADDING_MODE::REFLECTION => { gs_reflect(v, x_min, x_max) },
                                };
                                v
                            } else {
                                v
                            };

                            new_x.append(new_v);
                            i += 1;
                        };

                    let x = new_x.span();

                    let y = match mode {
                        MODE::NEAREST => {
                            pixel_at_ndarray(X_data, dims, X_data_stride, x, border, padding_mode)
                        },
                        MODE::LINEAR => {
                            gs_linear_interpolation_nd_with_x(
                                X_data, dims, X_data_stride, x, border, padding_mode
                            )
                        },
                        MODE::CUBIC => {
                            gs_cubic_interpolation_nd_with_x(
                                X_data, dims, X_data_stride, x, border, padding_mode
                            )
                        },
                    };

                    Y.append(y);
                    ix += 1;
                };

            c += 1;
        };

        n += 1;
    };

    TensorTrait::new(y_dims, Y.span())
}

fn gs_cubic_interpolation_1d_with_x<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +AddEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Rem<T>,
    +PrintTrait<T>,
>(
    data: Span<T>, x: T, border: Span<T>, padding_mode: PADDING_MODE
) -> T {
    let x_0 = NumberTrait::floor(x);
    let x_1 = x_0 + NumberTrait::one();
    let x_2 = x_1 + NumberTrait::one();
    let x_minus_1 = x_0 - NumberTrait::one();

    let coeffs = gs_get_cubic_coeffs(x - x_0);

    let v_0 = pixel_at_array(data, x_minus_1.try_into().unwrap(), border, padding_mode);
    let v_1 = pixel_at_array(data, x_0.try_into().unwrap(), border, padding_mode);
    let v_2 = pixel_at_array(data, x_1.try_into().unwrap(), border, padding_mode);
    let v_3 = pixel_at_array(data, x_2.try_into().unwrap(), border, padding_mode);

    let v: Span<T> = array![v_0, v_1, v_2, v_3].span();

    dot(coeffs, v)
}

fn gs_get_cubic_coeffs<
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
    x: T
) -> Span<T> {
    let one = NumberTrait::one();
    let two = one + NumberTrait::one();
    let three = two + NumberTrait::one();
    let four = three + NumberTrait::one();
    let five = four + NumberTrait::one();
    let eigth = four + four;

    let A = NumberTrait::neg(three / four);
    let x = NumberTrait::abs(x);

    let mut coeffs: Array<T> = array![];

    coeffs.append(((A * (x + one) - five * A) * (x + one) + eigth * A) * (x + one) - four * A);
    coeffs.append(((A + two) * x - (A + three)) * x * x + one);
    coeffs.append(((A + two) * (one - x) - (A + three)) * (one - x) * (one - x) + one);
    coeffs
        .append(
            ((A * ((one - x) + one) - five * A) * ((one - x) + one) + eigth * A) * ((one - x) + one)
                - four * A
        );

    coeffs.span()
}

fn gs_cubic_interpolation_nd_with_x<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +AddEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Rem<T>,
    +PrintTrait<T>,
>(
    data: Span<T>,
    data_dims: Span<usize>,
    data_stride: Span<usize>,
    x: Span<T>,
    border: Span<T>,
    padding_mode: PADDING_MODE
) -> T {
    let num_dims = data_dims.len();

    assert(num_dims == x.len(), 'pixel at nd array: wrong dim');
    assert(num_dims == (border.len() / 2), 'pixel at nd array: wrong dim');

    if num_dims == 1 {
        let a = gs_cubic_interpolation_1d_with_x(data, *x.at(0), border, padding_mode);
        return a;
    }

    let mut res1d: Array<T> = array![];

    let mut i = 0;
    while i != *data_dims
        .at(0) {
            let sub_data = SpanTrait::slice(data, i * *data_stride.at(0), *data_stride.at(0));
            let sub_x = SpanTrait::slice(x, 1, x.len() - 1);

            let data_dims_sub = SpanTrait::slice(data_dims, 1, data_dims.len() - 1);
            let data_stride_sub = SpanTrait::slice(data_stride, 1, data_stride.len() - 1);

            let border1 = SpanTrait::slice(border, 1, num_dims - 1);
            let border2 = SpanTrait::slice(border, num_dims + 1, num_dims - 1);
            let mut border = ArrayTrait::new();
            border.append_span(border1);
            border.append_span(border2);

            let r = gs_cubic_interpolation_nd_with_x(
                sub_data, data_dims_sub, data_stride_sub, sub_x, border.span(), padding_mode
            );

            res1d.append(r);
            i += 1;
        };

    gs_cubic_interpolation_1d_with_x(
        res1d.span(), *x.at(0), array![*border.at(0), *border.at(num_dims)].span(), padding_mode
    )
}

fn gs_get_linear_coeffs<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Sub<T>,>(
    x: T
) -> Span<T> {
    let x = NumberTrait::abs(x);

    array![NumberTrait::one() - x, x].span()
}

fn gs_linear_interpolation_1d_with_x<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Rem<T>,
    +PrintTrait<T>
>(
    data: Span<T>, x: T, border: Span<T>, padding_mode: PADDING_MODE
) -> T {
    let x_0 = NumberTrait::floor(x);
    let x_1 = x_0 + NumberTrait::one();

    let coeffs = gs_get_linear_coeffs(x - x_0);

    let v_0 = pixel_at_array(data, x_0.try_into().unwrap(), border, padding_mode);
    let v_1 = pixel_at_array(data, x_1.try_into().unwrap(), border, padding_mode);

    let v: Span<T> = array![v_0, v_1].span();

    dot(coeffs, v)
}

fn dot<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Add<T>, +TensorTrait<T>, +Mul<T>,>(
    a: Span<T>, b: Span<T>
) -> T {
    assert(a.len() == b.len(), 'dot: wrong len');

    let mut i = 0;
    let mut sum = NumberTrait::zero();
    while i != a.len() {
        sum = sum + *a.at(i) * *b.at(i);
        i += 1;
    };

    sum
}

fn gs_linear_interpolation_nd_with_x<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Rem<T>,
    +PrintTrait<T>
>(
    data: Span<T>,
    data_dims: Span<usize>,
    data_stride: Span<usize>,
    x: Span<T>,
    border: Span<T>,
    padding_mode: PADDING_MODE
) -> T {
    let num_dims = data_dims.len();

    assert(num_dims == x.len(), 'pixel at nd array: wrong dim');
    assert(num_dims == (border.len() / 2), 'pixel at nd array: wrong dim');

    if num_dims == 1 {
        let a = gs_linear_interpolation_1d_with_x(data, *x.at(0), border, padding_mode);
        return a;
    }

    let mut res1d: Array<T> = array![];

    let mut i = 0;
    while i != *data_dims
        .at(0) {
            let sub_data = SpanTrait::slice(data, i * *data_stride.at(0), *data_stride.at(0));
            let sub_x = SpanTrait::slice(x, 1, x.len() - 1);

            let data_dims_sub = SpanTrait::slice(data_dims, 1, data_dims.len() - 1);
            let data_stride_sub = SpanTrait::slice(data_stride, 1, data_stride.len() - 1);

            let border1 = SpanTrait::slice(border, 1, num_dims - 1);
            let border2 = SpanTrait::slice(border, num_dims + 1, num_dims - 1);
            let mut border = ArrayTrait::new();
            border.append_span(border1);
            border.append_span(border2);

            let r = gs_linear_interpolation_nd_with_x(
                sub_data, data_dims_sub, data_stride_sub, sub_x, border.span(), padding_mode
            );

            res1d.append(r);
            i += 1;
        };

    gs_linear_interpolation_1d_with_x(
        res1d.span(), *x.at(0), array![*border.at(0), *border.at(num_dims)].span(), padding_mode
    )
}

fn pixel_at_ndarray<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Rem<T>,
    +PrintTrait<T>,
>(
    ndarray: Span<T>,
    ndarray_dims: Span<usize>,
    ndarray_stride: Span<usize>,
    x: Span<T>,
    border: Span<T>,
    padding_mode: PADDING_MODE
) -> T {
    let num_dims = ndarray_dims.len();

    assert(num_dims == x.len(), 'pixel at nd array: wrong dim');
    assert(num_dims == (border.len() / 2), 'pixel at nd array: wrong dim');

    let i = *x.at(0);

    if num_dims == 1 {
        return pixel_at_array(ndarray, *x.at(0), border, padding_mode);
    }

    let d = NumberTrait::new_unscaled((*ndarray_dims.at(0)).into(), false);

    let ndarray = match padding_mode {
        PADDING_MODE::ZEROS => {
            let ndarray = if i >= NumberTrait::zero() && i < d {
                SpanTrait::slice(
                    ndarray, i.try_into().unwrap() * *ndarray_stride.at(0), *ndarray_stride.at(0)
                )
            } else {
                let ndarray: Span<T> = zeros(*ndarray_stride.at(0));
                ndarray
            };
            ndarray
        },
        PADDING_MODE::BORDER => {
            let i = clamp(i, NumberTrait::zero(), d - NumberTrait::one());
            SpanTrait::slice(
                ndarray, i.try_into().unwrap() * *ndarray_stride.at(0), *ndarray_stride.at(0)
            )
        },
        PADDING_MODE::REFLECTION => {
            let i: usize = (gs_reflect(i, *border.at(0), *border.at(num_dims))).try_into().unwrap();
            SpanTrait::slice(ndarray, i * *ndarray_stride.at(0), *ndarray_stride.at(0))
        },
    };

    let x = SpanTrait::slice(x, 1, x.len() - 1);
    let ndarray_dims = SpanTrait::slice(ndarray_dims, 1, ndarray_dims.len() - 1);
    let ndarray_stride = SpanTrait::slice(ndarray_stride, 1, ndarray_stride.len() - 1);

    let border1 = SpanTrait::slice(border, 1, num_dims - 1);
    let border2 = SpanTrait::slice(border, num_dims + 1, num_dims - 1);
    let mut border = ArrayTrait::new();
    border.append_span(border1);
    border.append_span(border2);

    pixel_at_ndarray(ndarray, ndarray_dims, ndarray_stride, x, border.span(), padding_mode)
}

fn pixel_at_array<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Rem<T>,
    +PrintTrait<T>,
>(
    array: Span<T>, i: T, border: Span<T>, padding_mode: PADDING_MODE
) -> T {
    let d = NumberTrait::new_unscaled(array.len().into(), false);

    let pixel = match padding_mode {
        PADDING_MODE::ZEROS => {
            let pixel = if i >= NumberTrait::zero() && i < d {
                *array.at(i.try_into().unwrap())
            } else {
                NumberTrait::zero()
            };
            pixel
        },
        PADDING_MODE::BORDER => {
            let i = clamp(i, NumberTrait::zero(), d - NumberTrait::one());
            let pixel = *array.at(i.try_into().unwrap());
            pixel
        },
        PADDING_MODE::REFLECTION => {
            let i: usize = (gs_reflect(i, *border.at(0), *border.at(1))).try_into().unwrap();
            let pixel = *array.at(i);
            pixel
        },
    };

    pixel
}

fn zeros<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>>(n: usize) -> Span<T> {
    let mut zeros: Array<T> = array![];
    let mut i = 0;
    while i != n {
        zeros.append(NumberTrait::zero());
        i += 1;
    };

    zeros.span()
}

fn rint<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +SubEq<T>,
    +Rem<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Add<T>,
    +Sub<T>
>(
    data: Span<T>
) -> Span<T> {
    // round to nearest if ties rounds to the nearest even value. 
    let mut rint: Array<T> = array![];
    let two: T = NumberTrait::one() + NumberTrait::one();

    let mut i = 0;
    while i != data
        .len() {
            let x = *data.at(i);
            let mut round = NumberTrait::round(x);

            let diff = round - x;
            if diff == NumberTrait::half() {
                if round % two != NumberTrait::zero() {
                    round -= NumberTrait::one()
                }
            }

            rint.append(round);
            i += 1;
        };

    rint.span()
}

fn clamp<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +PartialOrd<T>>(
    val: T, low: T, high: T
) -> T {
    if val < low {
        return low;
    }

    if val > high {
        return high;
    }

    val
}

fn gs_reflect<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Add<T>,
    +Sub<T>,
    +Div<T>,
    +Mul<T>,
    +Rem<T>,
    +PrintTrait<T>,
>(
    x: T, x_min: T, x_max: T
) -> T {
    let two: T = NumberTrait::one() + NumberTrait::one();
    let mut fx = x;
    let rng = x_max - x_min;

    let fx = if fx < x_min {
        let dx = x_min - fx;
        let n = NumberTrait::floor(dx / rng);
        let r = dx - n * rng;
        let fx = if NumberTrait::round(n % two) == NumberTrait::zero() {
            x_min + r
        } else {
            x_max - r
        };
        fx
    } else if fx > x_max {
        let dx = fx - x_max;
        let n = NumberTrait::floor(dx / rng);
        let r = dx - n * rng;

        let fx = if NumberTrait::round(n % two) == NumberTrait::zero() {
            x_max - r
        } else {
            x_min + r
        };
        fx
    } else {
        fx
    };

    fx
}

fn reverse<T, +Copy<T>, +Drop<T>,>(data: Span<T>) -> Span<T> {
    let mut rev: Array<T> = array![];
    let mut i = data.len();
    while i != 0 {
        rev.append(*data.at(i - 1));
        i -= 1;
    };

    rev.span()
}

fn get_sub<T, +Copy<T>, +Drop<T>,>(
    data: Span<T>, stride_data: Span<usize>, index: Span<usize>,
) -> Span<T> {
    let mut acc_indices = 0;
    let mut i = 0;
    while i != index.len() {
        acc_indices += *index.at(i) * *stride_data.at(i);
        i += 1;
    };

    SpanTrait::slice(data, acc_indices, *stride_data.at(index.len() - 1))
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

fn prepare_border<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +Into<usize, MAG>,
    +Neg<T>
>(
    self: @Tensor<T>, dims: Span<usize>, align_corner: usize
) -> Span<T> {
    let num_dims = dims.len();

    let mut borders1: Array<T> = array![];
    let mut borders2: Array<T> = array![];

    let mut i = 0;
    while i != num_dims {
        if align_corner == 0 {
            borders1.append(-NumberTrait::half());
            borders2
                .append(
                    NumberTrait::new_unscaled((*dims.at(i)).into(), false) - NumberTrait::half()
                );
        } else {
            borders1.append(NumberTrait::zero());
            borders2
                .append(
                    NumberTrait::new_unscaled((*dims.at(i)).into(), false) - NumberTrait::one()
                );
        }

        i += 1;
    };

    borders1.append_span(borders2.span());

    borders1.span()
}

fn arange(start: usize, end: usize, step: usize) -> Span<usize> {
    assert((end - start) % step == 0, 'incompatible step value');
    let mut arr: Array<usize> = array![];
    let mut i = start;
    while i != end {
        arr.append(i);
        i += step;
    };

    arr.span()
}

fn gs_denormalize_coordinates<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +Into<usize, MAG>
>(
    n: Span<T>, dims: Span<usize>, align_corner: usize
) -> Span<T> {
    let mut x: Array<T> = array![];

    let mut i = 0;
    while i != n
        .len() {
            let v = *n.at(i);
            let dim = *dims.at(i);
            x.append(gs_denormalize(v, dim, align_corner));
            i += 1;
        };

    x.span()
}

fn gs_denormalize<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +Sub<T>,
    +Into<usize, MAG>
>(
    n: T, length: usize, align_corner: usize
) -> T {
    let length = NumberTrait::new_unscaled(length.into(), false);
    let two: T = NumberTrait::one() + NumberTrait::one();

    let x = if align_corner == 0 {
        ((n + NumberTrait::one()) * length - NumberTrait::one()) / two
    } else {
        (n + NumberTrait::one()) / two * (length - NumberTrait::one())
    };

    x
}

fn get_all_coords(shape: Span<usize>) -> Span<Span<usize>> {
    let mut all_indices = array![];

    let mut i = 0;
    while i != shape.len() {
        all_indices.append(arange(0, *shape.at(i), 1));
        i += 1;
    };

    cartesian(all_indices.span())
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
    let mut output_arrays = ArrayTrait::<Array<usize>>::new();
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
        let mut x = ArrayTrait::new();
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
