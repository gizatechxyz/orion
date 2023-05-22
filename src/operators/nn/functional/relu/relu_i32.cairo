use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::utils::check_gas;

/// Cf: NNTrait::relu docstring
fn relu_i32(z: @Tensor<i32>, threshold: i32) -> Tensor<i32> {
    let mut data_result = ArrayTrait::<i32>::new();
    let mut data = *z.data;

    let zero = IntegerTrait::new(0, false);
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        if current_index < threshold {
            data_result.append(zero);
        } else {
            data_result.append(current_index);
        };
    };

    return TensorTrait::<i32>::new(*z.shape, data_result.span());
}
