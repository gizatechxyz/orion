use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::asin docstring
fn asin<
    T,
    MAG,
    impl TNumberTrait: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut self: Tensor<T>
) -> Tensor<T> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => { result.append((*item).asin()); },
            Option::None => { break; }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}
