use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::numbers::fixed_point::types::{Fixed, FixedType, MAX_u128};
use onnx_cairo::numbers::fixed_point::core::min;
use onnx_cairo::utils::check_gas;

/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<FixedType>) -> FixedType {
    let mut min_value: FixedType = Fixed::new(MAX_u128, false);

    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        let check_min = min(min_value, current_value);
        if (min_value > check_min) {
            min_value = check_min;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return min_value;
}
