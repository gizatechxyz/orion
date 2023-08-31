use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::helpers::check_compatibility;
use orion::utils::saturate;

/// Cf: PerfomanceTrait::dequantize_linear docstring
fn dequantize_linear<
    Q,
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl QIntoT: Into<Q, T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl QTensorIntoTTensor: Into<Tensor<Q>, Tensor<T>>,
    impl TDrop: Drop<T>,
    impl TCopy: Copy<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>
>(
    x: @Tensor<Q>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>
) -> Tensor::<T> {
    if (*x_scale.data).len() == 1 && (*x_zero_point.data).len() == 1 {
        dequantize_element_wise(*x, *x_scale.data[0], *x_zero_point.data[0])
    } else {
        check_compatibility(*x.shape, *x_scale.shape);
        check_compatibility(*x.shape, *x_zero_point.shape);
        check_compatibility(*x_scale.shape, *x_zero_point.shape);
        dequantize_per_axis(@(*x).into(), x_scale, x_zero_point)
    }
}

fn dequantize_per_axis<
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    x: @Tensor<T>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>
) -> Tensor::<T> {
    let mut result_data = ArrayTrait::<T>::new();

    (*x - *x_zero_point) * *x_scale
}

fn dequantize_element_wise<
    Q,
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl QIntoT: Into<Q, T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDrop: Drop<T>,
    impl TCopy: Copy<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>
>(
    mut x: Tensor::<Q>, x_scale: T, x_zero_point: T
) -> Tensor::<T> {
    let mut result_data = ArrayTrait::<T>::new();

    loop {
        match x.data.pop_front() {
            Option::Some(item) => {
                let dequantized = dequantize(*item, x_scale, x_zero_point);
                result_data.append(dequantized);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(x.shape, result_data.span(), x.extra);
}

fn dequantize<
    Q, T, impl QIntoT: Into<Q, T>, impl TSub: Sub<T>, impl TMul: Mul<T>, impl TDrop: Drop<T>
>(
    x: Q, x_scale: T, x_zero_point: T
) -> T {
    (x.into() - x_zero_point) * x_scale
}
