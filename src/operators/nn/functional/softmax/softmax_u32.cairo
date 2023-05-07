use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::operators::tensor::{tensor_u32, tensor_fp};
use onnx_cairo::numbers::fixed_point::core::FixedType;

/// Calculates the softmax function for a tensor of u32 values along the specified axis.
///
/// # Arguments
///
/// * `z` - A tensor of u32 values representing the input tensor.
/// * `axis` - The axis along which to compute the softmax function.
///
/// # Returns
///
/// * A tensor of fixed point numbers representing the result of applying the softmax function 
/// to the input tensor along the specified axis.
fn softmax_u32(z: @Tensor<u32>, axis: usize) -> Tensor<FixedType> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;

    return softmax;
}
