use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;
use option::OptionTrait;

use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::tensor_u32;
use onnx_cairo::utils::check_gas;

fn relu(z: @Tensor<u32>) -> Tensor<u32> {
    let mut data_result = ArrayTrait::<u32>::new();
    let mut data = *z.data;

    let zero = 0;
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

    return TensorTrait::<u32>::new(*z.shape, data_result.span());
}
