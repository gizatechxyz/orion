use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};

/// Cf: TensorTrait::reduce_sum docstring
fn reduce_l1<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, axis: usize, keepdims: bool
) -> Tensor<T> {
    let data_abs = self.abs();

    data_abs
        .reduce_sum(
            Option::Some(array![axis.try_into().unwrap()].span()),
            Option::Some(keepdims),
            Option::Some(false)
        )
}
