use core::traits::TryInto;
use core::traits::Into;
use core::traits::Copy;
use core::traits::Drop;
use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use onnx_cairo::numbers::fixed_point::types::{FixedType,Fixed,ONE_u128};
use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait};
use onnx_cairo::operators::tensor::implementations::{impl_tensor_u32,impl_tensor_fp};
use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::utils::check_gas;


/// Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given u32 tensor.
///
/// The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.
///
/// # Arguments
/// * `z` - A snapshot of a u32 tensor to which the Leaky ReLU function will be applied.
/// * `alpha` - A snapshot of a FixedType scalar that defines the alpha value of the Leaky ReLU function.
/// * `threshold` - a u32 scalar that defines the threshold below which the alpha value is applied.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A new FixedType tensor with the same shape as the input tensor and the Leaky ReLU function
///   applied element-wise.

fn leaky_relu_u32(z: @Tensor<u32>, alpha: @FixedType, threshold:u32) -> Tensor<FixedType> {
    assert(*alpha.mag < ONE_u128, 'alpha must be less than 1_fp');

    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index = Fixed::new_unscaled(current_index.into(), false);
        if  current_index >= threshold {
            data_result.append(fp_current_index);
        } else {
            data_result.append(fp_current_index * *alpha);
        };
    };

    return TensorTrait::<FixedType>::new(*z.shape, data_result.span());

}




   
    