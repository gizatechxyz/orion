use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::nn::nn_i32::NN;

#[test]
#[available_gas(2000000)]
fn relu_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(1_u32, false);
    let val_2 = IntegerTrait::new(2_u32, false);
    let val_3 = IntegerTrait::new(1_u32, true);
    let val_4 = IntegerTrait::new(2_u32, true);

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);

    let mut tensor = TensorTrait::new(shape.span(), data.span());
    let mut result = NN::relu(@tensor);

    let data_0 = *result.data.at(0);
    assert(data_0.mag == 1_u32, 'result[0] == 1');
    assert(data_0.sign == false, 'result[0] == 1');

    let data_3 = *result.data.at(3);
    assert(data_3.mag == 0_u32, 'result[3] == 0');
    assert(data_3.sign == false, 'result[3] == 0');
}
