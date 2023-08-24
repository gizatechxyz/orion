use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, MAX, FP8x23PartialOrd};
use orion::numbers::fixed_point::implementations::fp8x23::math::comp::min;


/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<FixedType>) -> FixedType {
    let mut min_value: FixedType = FixedTrait::new(MAX - 1, false);

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                let check_min = min(min_value, *item);
                if (min_value > check_min) {
                    min_value = check_min;
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return min_value;
}
