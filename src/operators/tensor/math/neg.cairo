use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::neg docstring
fn neg<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumberTrait: NumberTrait<T, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result: Array<T> = array![];
    loop {
        match z.data.pop_front() {
            Option::Some(item) => { data_result.append((*item).neg()); },
            Option::None => { break; }
        };
    };

    TensorTrait::<T>::new(z.shape, data_result.span())
}
