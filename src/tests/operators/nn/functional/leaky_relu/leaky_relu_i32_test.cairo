use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::nn::nn_i32::NN;
use onnx_cairo::numbers::fixed_point::types::{FixedType,Fixed,ONE_u128};

#[test]
#[available_gas(2000000)]
fn leaky_relu_i32_test() { 
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(1_u32, false);
    let val_2 = IntegerTrait::new(2_u32, false);
    let val_3 = IntegerTrait::new(1_u32, true);
    let val_4 = IntegerTrait::new(2_u32, true);
    let val_5 = IntegerTrait::new(0_u32, false);
    let val_6 = IntegerTrait::new(0_u32, false);

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);
    data.append(val_5);
    data.append(val_6);

    let mut tensor = TensorTrait::new(shape.span(), data.span());
    let alpha = Fixed::new(6710886_u128, false); // 0.1

    let mut result = NN::leaky_relu(@tensor, @alpha);

    let data_0 = *result.data.at(0);
    assert(data_0.mag == ONE_u128, 'result[0] == 67108864'); // 1
    assert(data_0.sign == false, 'result[0].sign == false');

    let data_3 = *result.data.at(3);
    assert(data_3.mag == 13421772, 'result[3] == 113421772');// 2 * 0.1 = 0.2
    assert(data_3.sign == true, 'result[3].sign == true');

    let data_5 = *result.data.at(5);
    assert(data_5.mag == 0, 'result[5] == 0');
    assert(data_5.sign == false, 'result[5].sign == false');
}

