use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::utils::check_gas;

/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<i32>) -> i32 {
    let mut min_value: i32 = IntegerTrait::new(2147483647_u32, false);

    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        let check_min = min_value.min(current_value);
        if (min_value > check_min) {
            min_value = check_min;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return min_value;
}
