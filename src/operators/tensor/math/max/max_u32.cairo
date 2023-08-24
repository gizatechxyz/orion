use array::SpanTrait;
use option::OptionTrait;


/// Cf: TensorTrait::max docstring
fn max_in_tensor(mut vec: Span::<u32>) -> u32 {
    let mut max_value = 0;

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                if (max_value < *item) {
                    max_value = *item;
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return max_value;
}
