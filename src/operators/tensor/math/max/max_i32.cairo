use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};


/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<i32>) -> i32 {
    let mut max_value: i32 = IntegerTrait::new(2147483647, true);

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                let check_max = max_value.max(*item);
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
