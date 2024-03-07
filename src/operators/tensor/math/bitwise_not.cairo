use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index};
use orion::operators::tensor::helpers::{
    broadcast_shape, broadcast_index_mapping, len_from_shape
};

/// Cf: TensorTrait::and docstring
fn bitwise_not<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>
) -> Tensor<T> {
    let mut result: Array<T> = array![];
    let num_elements = len_from_shape(*self.shape);
    let mut n: usize = 0;
    while n != num_elements {
        result.append(NumberTrait::bitwise_not(*(*self.data)[n]));
        n += 1;
    };

    TensorTrait::<T>::new(*self.shape, result.span())


}
