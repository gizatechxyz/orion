use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::operators::tensor::tensor_fp;
use onnx_cairo::operators::math::fixed_point::core::FixedType;
use onnx_cairo::utils::check_gas;

/// Calculates the softmax function for a tensor of i32 values along the specified axis.
///
/// # Arguments
///
/// * `z` - A tensor of i32 values representing the input tensor.
/// * `axis` - The axis along which to compute the softmax function.
///
/// # Returns
///
/// * A tensor of fixed point numbers representing the result of applying the softmax function 
/// to the input tensor along the specified axis.
fn softmax_i32(z: @Tensor<i32>, axis: usize) -> Tensor<FixedType> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis);
    let softmax = exp_tensor / sum;

    return softmax;
}
