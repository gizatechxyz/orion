use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::math::reduce_sum_single_axis::reduce_sum_single_axis;

/// Cf: TensorTrait::reduce_sum_square docstring
fn reduce_log_sum<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, axis: usize, keepdims: bool
) -> Tensor<T> {
    let tensor_square_sum = reduce_sum_single_axis(self, axis: axis, keepdims: keepdims);
    let tensor_square_sum_log = tensor_square_sum.log();

    tensor_square_sum_log
}
