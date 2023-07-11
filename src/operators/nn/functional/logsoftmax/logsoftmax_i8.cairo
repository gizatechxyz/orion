use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{
    impl_tensor_i8::Tensor_i8, impl_tensor_fp::{Tensor_fp, FixedTypeTensorDiv}
};
use orion::numbers::fixed_point::core::FixedType;

/// Cf: NNTrait::logsoftmax docstring
fn logsoftmax_i8(z: @Tensor<i8>, axis: usize) -> Tensor<FixedType> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;
    let logsoftmax = softmax.ln();

    return logsoftmax;
}
