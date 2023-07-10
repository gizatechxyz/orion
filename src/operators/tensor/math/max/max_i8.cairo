use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::utils::check_gas;

/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<i8>) -> i8 {
    let mut max_value: i8 = IntegerTrait::new(128, true);

    loop {
        check_gas();

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
