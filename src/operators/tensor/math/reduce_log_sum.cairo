use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};

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
    let tensor_square_sum = self.reduce_sum(Option::Some(array![axis].span()), Option::Some(keepdims), Option::Some(false));
    let tensor_square_sum_log = tensor_square_sum.log();

    tensor_square_sum_log
}
