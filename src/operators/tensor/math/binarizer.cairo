use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::binarizer docstring
fn binarizer<
    T,
    MAG,
    impl TTensor: TensorTrait<usize>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut self: Tensor<T>, threshold: @T
) -> Tensor<usize> {
    let mut binarized_data: Array<usize> = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                if (*item) > (*threshold) {
                    binarized_data.append(1);
                } else {
                    binarized_data.append(0);
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(self.shape, binarized_data.span());
}
