use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::operators::tensor::implementations::{impl_tensor_u32, impl_tensor_fp};
use onnx_cairo::numbers::fixed_point::core::FixedType;

/// Cf: NNTrait::softmax docstring
fn softmax_u32(z: @Tensor<u32>, axis: usize) -> Tensor<FixedType> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;

    return softmax;
}
