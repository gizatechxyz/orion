use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::implementations::impl_tensor_u32;
use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait};
use onnx_cairo::operators::nn::core::NNTrait;
use onnx_cairo::operators::nn::implementations::impl_nn_u32;
use onnx_cairo::numbers::fixed_point::types::{FixedType, Fixed, ONE_u128};

#[test]
#[available_gas(2000000)]
fn leaky_relu_u32_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::<u32>::new();
    let val_1 = 4_u32;
    let val_2 = 3_u32;
    let val_3 = 2_u32;
    let val_4 = 1_u32;
    let val_5 = 0_u32;
    let val_6 = 0_u32;

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);
    data.append(val_5);
    data.append(val_6);

    let mut tensor = TensorTrait::new(shape.span(), data.span());
    let alpha = Fixed::new(6710886, false); // 0.1
    let threshold = 3_u32;
    let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

    let data_0 = *result.data.at(0);
    assert(data_0 == Fixed::new(268435456, false), 'result[0] == 268435456'); // 4 

    let data_1 = *result.data.at(1);
    assert(data_1 == Fixed::new(201326592, false), 'result[1] == 20132658'); // 3

    let data_3 = *result.data.at(3);
    assert(data_3 == Fixed::new(6710886, false), 'result[3] == 6710886'); // 0.1

    let data_5 = *result.data.at(5);
    assert(data_5 == Fixed::new(0, false), 'result[5] == 0');
}

