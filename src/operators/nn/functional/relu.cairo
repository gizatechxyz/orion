use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::relu docstring
fn relu<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result = ArrayTrait::<T>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if (*item) < NumberTrait::zero() {
                    data_result.append(NumberTrait::zero());
                } else {
                    data_result.append(*item);
                };
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}
