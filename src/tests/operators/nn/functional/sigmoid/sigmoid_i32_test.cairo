use array::ArrayTrait;
use core::debug::PrintTrait;
use array::SpanTrait;

use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32;
use orion::numbers::fixed_point::types::Fixed;



#[test]
#[available_gas(5000000)]
fn sigmoid_i32_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(0_u32, false);
    let val_2 = IntegerTrait::new(1_u32, false);
    let val_3 = IntegerTrait::new(2_u32, true);
    let val_4 = IntegerTrait::new(254_u32, false);

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);

    let mut tensor = TensorTrait::new(shape.span(), data.span());
    let mut result = NNTrait::sigmoid(@tensor);

    let data_0 = *result.data.at(0);
    assert(data_0 == Fixed::new(4194304, false), 'result[0] == 4194304'); // 0.5

    let data_1 = *result.data.at(1);
    assert(data_1 == Fixed::new(6132564, false), 'result[1] == 6132564'); // 0.7310586

    let data_2 = *result.data.at(2);
    assert(data_2 == Fixed::new(999946, false), 'result[2] == 999946'); // 0.11920285

    let data_3 = *result.data.at(3);
    assert(data_3 == Fixed::new(8388608, false), 'result[3] == 8388608'); // 1
}

