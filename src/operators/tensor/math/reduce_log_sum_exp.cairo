use core::option::OptionTrait;
use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::debug::PrintTrait; 

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::numbers::fixed_point::core::FixedTrait;
    
/// Cf: TensorTrait::reduce_log_sum_exp docstring
fn reduce_log_sum_exp<
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
    let tensor_exp = self.exp();
    let tensor_exp_sum = tensor_exp.reduce_sum(axis: axis, keepdims: keepdims);
    let tensor_exp_sum_log = tensor_exp_sum.log();

    return tensor_exp_sum_log;
}


