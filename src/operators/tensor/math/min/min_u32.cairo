use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;


/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<u32>) -> u32 {
    let mut min_value = 4294967295;

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                if (min_value > *item) {
                    min_value = *item;
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return min_value;
}
