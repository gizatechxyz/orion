use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::numbers::fixed_point::types::{Fixed, FixedType, MAX_u128};
use onnx_cairo::numbers::fixed_point::core::max;
use onnx_cairo::utils::check_gas;

/// Finds the maximum value in a `Tensor<FixedType>` array.
///
/// # Arguments
/// * `vec` -  A span containing the data array of FixedType elements.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An FixedType value representing the maximum value in the array.
fn max_in_tensor(mut vec: Span::<FixedType>) -> FixedType {
    let mut max_value: FixedType = Fixed::new(MAX_u128, true);

    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        let check_max = max(max_value, current_value);
        if (max_value < check_max) {
            max_value = check_max;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return max_value;
}
