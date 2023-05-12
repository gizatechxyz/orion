use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::tensor::implementations::impl_tensor_u32;
use onnx_cairo::utils::check_gas;

/// Cf: TensorTrait::min docstring
fn min_in_tensor(mut vec: Span::<u32>) -> u32 {
    let mut min_value = 4294967295_u32;
    loop {
        check_gas();

        let current_value = *vec.pop_front().unwrap();

        if (min_value > current_value) {
            min_value = current_value;
        }

        if vec.len() == 0 {
            break ();
        };
    };

    return min_value;
}
