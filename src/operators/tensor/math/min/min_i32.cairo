use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};


/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<i32>) -> i32 {
    let mut min_value: i32 = IntegerTrait::new(2147483647, false);

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                let check_min = min_value.min(*item);
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

