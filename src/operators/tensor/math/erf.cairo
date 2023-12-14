use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;


/// Cf: TensorTrait::erf docstring
fn erf<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TFixed: FixedTrait<T, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result = ArrayTrait::<T>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => { data_result.append((*item).erf()); },
            Option::None(_) => { break; }
        };
    };

    return TensorTrait::<T>::new(z.shape, data_result.span());
}
