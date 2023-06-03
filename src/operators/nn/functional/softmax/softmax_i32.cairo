use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{impl_tensor_i32, impl_tensor_fp8x23};
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::numbers::fixed_point::core::FixedType;

/// Cf: NNTrait::softmax docstring
fn softmax_i32_fp8x23(z: @Tensor<i32>, axis: usize) -> Tensor<FixedType<fp8x23>> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;

    return softmax;
}

