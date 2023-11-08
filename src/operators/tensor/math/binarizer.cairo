use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::binarizer docstring
fn binarizer<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut self: Tensor<T>, threshold: @T
) -> Tensor<T> {
    let mut binarized_data = ArrayTrait::<T>::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                if (*item) > (*threshold) {
                    binarized_data.append(NumberTrait::one());
                } else {
                    binarized_data.append(NumberTrait::zero());
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(self.shape, binarized_data.span());
}
