use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::tanh docstring
fn tanh<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl FTensor: TensorTrait<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    mut self: Tensor<T>
) -> Tensor<T> {
    let mut result = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                result.append((*item).tanh());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(self.shape, result.span());
}
