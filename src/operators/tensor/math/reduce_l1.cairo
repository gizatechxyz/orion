use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::math::reduce_sum::reduce_sum_single_axis;

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

    reduce_sum_single_axis(@data_abs, axis: axis, keepdims: keepdims)
}
