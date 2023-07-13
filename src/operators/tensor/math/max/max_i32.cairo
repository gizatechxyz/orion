use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};


/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<i32>) -> i32 {
    let mut max_value: i32 = IntegerTrait::new(2147483647, true);

    loop {
        let current_value = *vec.pop_front().unwrap();

        let check_max = max_value.max(current_value);
        if (max_value < check_max) {
            max_value = check_max;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return max_value;
}
