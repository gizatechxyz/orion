use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, MAX, FP16x16PartialOrd};
use orion::numbers::fixed_point::math::math_16x16::max;


/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<FixedType>) -> FixedType {
    let mut max_value: FixedType = FixedTrait::new(MAX, true);

    loop {


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
