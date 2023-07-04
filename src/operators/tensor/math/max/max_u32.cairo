use array::SpanTrait;
use option::OptionTrait;



/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<u32>) -> u32 {
    let mut max_value = 0;
    loop {


        let current_value = *vec.pop_front().unwrap();

        if (max_value < current_value) {
            max_value = current_value;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return max_value;
}
