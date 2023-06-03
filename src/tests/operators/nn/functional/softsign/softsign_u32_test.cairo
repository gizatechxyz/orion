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
fn softsign_u32_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::<u32>::new();
    let val_1 = 0_u32;
    let val_2 = 1_u32;
    let val_3 = 2_u32;
    let val_4 = 3_u32;

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);

    let mut tensor = TensorTrait::new(shape.span(), data.span());
    let mut result = NNTrait::softsign(@tensor);

    let data_0 = *result.data.at(0);
    assert(data_0 == FixedTrait::new(0, false), 'result[0] == 0'); // 0 

    let data_1 = *result.data.at(1);
    assert(data_1 == FixedTrait::new(4194304, false), 'result[1] == 4194304'); // 0.5

    let data_2 = *result.data.at(2);
    assert(data_2 == FixedTrait::new(5592405, false), 'result[2] == 5592405'); // 0.67

    let data_3 = *result.data.at(3);
    assert(data_3 == FixedTrait::new(6291456, false), 'result[3] == 6291456'); // 0.75
}

