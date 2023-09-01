use core::debug::PrintTrait;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::TryInto;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::helpers::check_compatibility;
use orion::operators::tensor::math::arithmetic::saturated_add;
use orion::utils::saturate;

/// Cf: TensorTrait::quantize_linear docstring
fn quantize_linear<
    T,
    F,
    Q,
    impl TTensor: TensorTrait<T, F>,
    impl QTensor: TensorTrait<Q, F>,
    impl TAdd: Add<T>,
    impl TDiv: Div<T>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TTryInto: TryInto<T, Q>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>,
>(
    x: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>, min: T, max: T
) -> Tensor::<Q> {
    if (*y_scale.data).len() == 1 && (*y_zero_point.data).len() == 1 {
        quantize_element_wise(*x, *y_scale.data[0], *y_zero_point.data[0], min, max)
    } else {
        check_compatibility(*x.shape, *y_scale.shape);
        check_compatibility(*x.shape, *y_zero_point.shape);
        check_compatibility(*y_scale.shape, *y_zero_point.shape);
        quantize_per_axis(x, y_scale, y_zero_point, min, max)
    }
}

fn quantize_per_axis<
    T,
    F,
    Q,
    impl TTensor: TensorTrait<T, F>,
    impl QTensor: TensorTrait<Q, F>,
    impl TAdd: Add<T>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TTryInto: TryInto<T, Q>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QDrop: Drop<Q>,
>(
    x: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>, min: T, max: T
) -> Tensor::<Q> {
    saturated_add(@(*x / *y_scale), y_zero_point, min, max)
}

fn quantize_element_wise<
    T,
    F,
    Q,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>,
    impl TPartialOrd: PartialOrd<T>,
    impl TDiv: Div<T>,
    impl TAdd: Add<T>,
    impl TTryIntoQ: TryInto<T, Q>,
    impl QTensor: TensorTrait<Q, F>
>(
    mut x: Tensor::<T>, y_scale: T, y_zero_point: T, min: T, max: T
) -> Tensor::<Q> {
    let mut result_data = ArrayTrait::<Q>::new();

    loop {
        match x.data.pop_front() {
            Option::Some(item) => {
                let quantized = quantize(*item, y_scale, y_zero_point, min, max);
                result_data.append(quantized);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(x.shape, result_data.span());
}

fn quantize<
    T,
    Q,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TDiv: Div<T>,
    impl TAdd: Add<T>,
    impl TTryIntoQ: TryInto<T, Q>
>(
    x: T, y_scale: T, y_zero_point: T, min: T, max: T
) -> Q {
    saturate(min, max, ((x / y_scale) + y_zero_point)).try_into().unwrap()
}
