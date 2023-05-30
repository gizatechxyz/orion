use core::debug::PrintTrait;
use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_u32;
use orion::numbers::fixed_point::types::Fixed;

#[test]
#[available_gas(5000000)]
fn sigmoid_u32_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::<u32>::new();
    let val_1 = 0_u32;
    let val_2 = 1_u32;
    let val_3 = 2_u32;
    let val_4 = 254_u32;

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
    assert(data_2 == Fixed::new(7388661, false), 'result[2] == 7388661'); // 0.88079703

    let data_3 = *result.data.at(3);
    assert(data_3 == Fixed::new(8388608, false), 'result[3] == 8388608'); // 1
}

