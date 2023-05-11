use array::SpanTrait;

use onnx_cairo::numbers::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::performance::performance_i32::performance::quantize_linear;

/// Performs a linear transformation of the input tensor using the provided weights and bias.
///
/// # Arguments
/// * `z` - A 1D tensor of i32 values representing the input tensor.
/// * `weights` - A 2D tensor of i32 values representing the weights for the linear transformation.
/// * `bias` - A 1D tensor of i32 values representing the bias for the linear transformation.
/// * `quantized` - A boolean flag indicating whether or not to quantize the result of the linear transformation.
///
/// # Panics
/// This function asserts that the input tensor `z` must be 1D, weights tensor must be 2D, and bias tensor must be 1D.
///
/// # Returns
/// * A tensor of i32 values representing the result of the linear transformation, possibly quantized.
fn linear_i32(
    z: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>, quantized: bool
) -> Tensor<i32> {
    assert(z.shape.len() == 1, 'input tensor must be 1D');
    assert(weights.shape.len() == 2, 'weights tensor must be 2D');
    assert(bias.shape.len() == 1, 'bias tensor must be 1D');

    let dot = weights.matmul(@z);
    let sum = dot + bias;

    if quantized {
        return quantize_linear(@sum);
    }

    return sum;
}
