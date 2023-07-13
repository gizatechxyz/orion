use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, MAX, FP16x16PartialOrd};
use orion::numbers::fixed_point::math::math_16x16::min;


/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<FixedType>) -> FixedType {
    let mut min_value: FixedType = FixedTrait::new(MAX - 1, false);

    loop {
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
