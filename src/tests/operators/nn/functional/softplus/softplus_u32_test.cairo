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
fn softplus_u32_test() {
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
    let mut result = NNTrait::softplus(@tensor);

    let data_0 = *result.data.at(0);
    assert(data_0 == Fixed::new(5814556, false), 'result[0] == 5814556'); // 0.6931452

    let data_1 = *result.data.at(1);
    assert(data_1 == Fixed::new(11016447, false), 'result[1] == 11016447'); // 1.31326096

    let data_2 = *result.data.at(2);
    assert(data_2 == Fixed::new(17841964, false), 'result[2] == 17841964'); // 2.12692796

    let data_3 = *result.data.at(3);
    assert(data_3 == Fixed::new(25573406, false), 'result[3] == 25573406'); // 3.04858728
}

