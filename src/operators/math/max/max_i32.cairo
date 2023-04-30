use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32Impl;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::utils::check_gas;

/// Finds the maximum value in a `Tensor<i32>` array.
///
/// # Arguments
/// * `vec` -  A span containing the data array of i32 elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 value representing the maximum value in the array.
fn max_in_tensor(mut vec: Span::<i32>) -> i32 {
    let mut max_value: i32 = IntegerTrait::new(0_u32, false);

    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        let check_max = max_value.max(current_value);
        if (max_value < check_max) {
            max_value = check_max;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return max_value;
}
