use core::array::ArrayTrait;
use option::OptionTrait;
use core::array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::optional_get_element docstring
fn optional_get_element<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumberTrait: NumberTrait<T, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut z: Tensor<T>, index: usize
) -> Tensor<T> {
    let mut data_result = ArrayTrait::<T>::new();

    // use of match to get element within and out the array bound
    match z.data.get(index) {
        Option::Some(item) => { data_result.append((*item.unbox())); },
        Option::None => {}
    };

    return TensorTrait::<T>::new(z.shape, data_result.span());
}
