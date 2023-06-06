use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_16x16;
use orion::numbers::fixed_point::math::math_16x16::max;
use orion::utils::check_gas;

/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<FixedType>) -> FixedType {
    let mut max_value: FixedType = FixedTrait::new(impl_16x16::MAX, true);

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
