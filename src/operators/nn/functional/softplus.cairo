use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;


/// Cf: NNTrait::softplus docstring
fn softplus<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TFixed: FixedTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAdd: Add<T>,
    impl TDiv: Div<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result = ArrayTrait::<T>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let result = (FixedTrait::ONE() + (*item).exp()).ln();
                data_result.append(result);
            },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}
