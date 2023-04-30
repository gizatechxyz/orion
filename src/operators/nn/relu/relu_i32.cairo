use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::utils::check_gas;

/// Applies the rectified linear unit (ReLU) activation function element-wise to a given i32 tensor.
///
/// The ReLU function is defined as f(x) = max(0, x), where x is the input element.
///
/// # Arguments
/// * `z` - A reference to an i32 tensor to which the ReLU function will be applied.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A new i32 tensor with the same shape as the input tensor and the ReLU function
///   applied element-wise.
fn relu(z: @Tensor<i32>) -> Tensor<i32> {
    let mut data_result = ArrayTrait::<i32>::new();
    let mut data = *z.data;

    let zero = IntegerTrait::<i32>::new(0, false);
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        if current_index > zero {
            data_result.append(current_index);
        } else {
            data_result.append(zero);
        };
    };

    return TensorTrait::<i32>::new(*z.shape, data_result.span());
}
