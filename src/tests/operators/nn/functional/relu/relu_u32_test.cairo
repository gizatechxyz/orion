use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::implementations::impl_tensor_u32;
use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait};
use onnx_cairo::operators::nn::core::NNTrait;
use onnx_cairo::operators::nn::implementations::impl_nn_u32;

#[test]
#[available_gas(2000000)]
fn relu_u32_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::<u32>::new();
    let val_1 = 1_u32;
    let val_2 = 2_u32;
    let val_3 = 3_u32;
    let val_4 = 4_u32;

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);

    let mut tensor = TensorTrait::new(shape.span(), data.span());
    let threshold = 3_u32;
    let mut result = NNTrait::relu(@tensor, threshold);

    let data_0 = *result.data.at(0);
    assert(data_0 == 0, 'result[0] == 0');

    let data_1 = *result.data.at(1);
    assert(data_1 == 0, 'result[1] == 0');

    let data_2 = *result.data.at(2);
    assert(data_2 == 3, 'result[2] == 3');

    let data_3 = *result.data.at(3);
    assert(data_3 == 4, 'result[3] == 4');
}
