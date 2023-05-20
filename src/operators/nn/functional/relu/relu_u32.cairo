use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::operators::tensor::implementations::impl_tensor_u32;
use onnx_cairo::utils::check_gas;

/// Applies the rectified linear unit (ReLU) activation function element-wise to a given u32 tensor.
///
/// The ReLU function is defined as f(x) = max(0, x), where x is the input element.
///
/// # Arguments
/// * `z` - A reference to an u32 tensor to which the ReLU function will be applied.
/// * `threshold` - a u32 scalar that defines the threshold below which the Relu function returns 0.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A new u32 tensor with the same shape as the input tensor and the ReLU function
///   applied element-wise.
fn relu_u32(z: @Tensor<u32>, threshold:u32 ) -> Tensor<u32> {
    let mut data_result = ArrayTrait::<u32>::new();
    let mut data = *z.data;

    let zero = 0;
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        if current_index < threshold {
            data_result.append(zero);
        } else {
            data_result.append(current_index);
        };
    };

    return TensorTrait::<u32>::new(*z.shape, data_result.span());
}
