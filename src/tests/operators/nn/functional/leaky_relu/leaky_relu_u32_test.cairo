use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_u32;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::numbers::fixed_point::implementations::impl_8x23;

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
    let alpha = FixedTrait::new(838861, false); // 0.1
    let threshold = 3_u32;
    let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

    let data_0 = *result.data.at(0);
    assert(data_0 == FixedTrait::new(33554432, false), 'result[0] == 33554432'); // 4 

    let data_1 = *result.data.at(1);
    assert(data_1 == FixedTrait::new(25165824, false), 'result[1] == 25165824'); // 3

    let data_3 = *result.data.at(3);
    assert(data_3 == FixedTrait::new(838861, false), 'result[3] == 838861'); // 0.1

    let data_5 = *result.data.at(5);
    assert(data_5 == FixedTrait::new(0, false), 'result[5] == 0');
}

