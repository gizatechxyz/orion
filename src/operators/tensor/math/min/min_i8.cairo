use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};


/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<i8>) -> i8 {
    let mut min_value: i8 = IntegerTrait::new(127, false);

    loop {
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
