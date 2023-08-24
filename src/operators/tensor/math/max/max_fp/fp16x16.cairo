use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, MAX, FP16x16PartialOrd
};
use orion::numbers::fixed_point::implementations::fp16x16::math::comp::max;

/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<FixedType>) -> FixedType {
    let mut max_value: FixedType = FixedTrait::new(MAX, true);

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                let check_max = max(max_value, *item);
                if (max_value < check_max) {
                    max_value = check_max;
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return max_value;
}


