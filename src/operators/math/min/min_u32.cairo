use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::tensor::tensor_u32;
use onnx_cairo::utils::check_gas;

/// Finds the minimum value in a `Tensor<u32>` array.
///
/// # Arguments
/// * `vec` -  A span containing the data array of u32 elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An u32 value representing the minimum value in the array.
fn min_in_tensor(mut vec: Span::<u32>) -> u32 {
    let mut min_value = 4294967295_u32;
    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        if (min_value > current_value) {
            min_value = current_value;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return min_value;
}