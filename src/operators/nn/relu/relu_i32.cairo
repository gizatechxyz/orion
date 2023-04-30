use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::utils::check_gas;

fn relu(z: @Tensor<i32>) -> Tensor<i32> {
    let mut data_result = ArrayTrait::<i32>::new();
    let mut data = *z.data;

    let zero = IntegerTrait::<i32>::new(0, false);
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        if current_index > zero {
            data_result.append(current_index);
        } else {
            data_result.append(zero);
        };
    };

    return TensorTrait::<i32>::new(*z.shape, data_result.span());
}
