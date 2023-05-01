use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::tensor::tensor_u32;
use onnx_cairo::utils::check_gas;

/// Finds the maximum value in a `Tensor<u32>` array.
///
/// # Arguments
/// * `vec` -  A span containing the data array u32 elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An u32 value representing the maximum value in the array.
fn max_in_tensor(mut vec: Span::<u32>) -> u32 {
    let mut max_value = 0_u32;
    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        if (max_value < current_value) {
            max_value = current_value;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return max_value;
}
