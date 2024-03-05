use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

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
    let mut data_result: Array<T> = array![];

    // use of match to get element within and out the array bound
    match z.data.get(index) {
        Option::Some(item) => { data_result.append((*item.unbox())); },
        Option::None => {}
    };

    TensorTrait::<T>::new(z.shape, data_result.span())
}
