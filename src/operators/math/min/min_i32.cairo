use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32Impl;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::utils::check_gas;

/// Finds the minimum value in a `Tensor<i32>` array.
///
/// # Arguments
/// * `vec` -  A span containing the data array of i32 elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An i32 value representing the minimum value in the array.
fn min_in_tensor(mut vec: Span::<i32>) -> i32 {
    let mut min_value: i32 = IntegerTrait::new(2147483647_u32, false);

    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        let check_min = min_value.min(current_value);
        if (min_value > check_min) {
            min_value = check_min;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return min_value;
}
