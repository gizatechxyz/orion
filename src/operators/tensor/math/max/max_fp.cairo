use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::types::{Fixed, FixedType, MAX_u128};
use orion::numbers::fixed_point::core::max;
use orion::utils::check_gas;

/// Cf: TensorTrait::max docstring
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
