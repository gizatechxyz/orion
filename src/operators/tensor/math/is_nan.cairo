use core::array::ArrayTrait;
use core::option::OptionTrait;
use core::array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::tensor_bool::BoolTensor;

/// Cf: TensorTrait::is_nan docstring
fn is_nan<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    x: @Tensor<T>
) -> Tensor<bool> {
    let mut data_result = ArrayTrait::<bool>::new();
    let mut y: Span<T> = *x.data;
    loop {
        match y.pop_front() {
            Option::Some(item) => { data_result.append((*item).is_nan()); },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(*x.shape, data_result.span());
}
