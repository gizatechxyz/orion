use core::option::OptionTrait;
use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::fixed_point::core::FixedImpl;


#[test]
#[available_gas(5000000)]
fn softplus_i32_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(0_u32, false);
    let val_2 = IntegerTrait::new(1_u32, false);
    let val_3 = IntegerTrait::new(2_u32, true);
    let val_4 = IntegerTrait::new(3_u32, true);

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
    let mut result = NNTrait::softplus(@tensor);

    let data_0 = *result.data.at(0);
    assert(data_0.mag == 5814556, 'result[0] == 5814556'); // 0.6931452
    assert(data_0.sign == false, 'result[0].sign == false');

    let data_1 = *result.data.at(1);
    assert(data_1.mag == 11016447, 'result[1] == 11016447'); // 1.31326096
    assert(data_1.sign == false, 'result[1].sign == false');

    let data_2 = *result.data.at(2);
    assert(data_2.mag == 1064751, 'result[2] == 1064751'); // 0.12692797
    assert(data_2.sign == false, 'result[2].sign == false');

    let data_3 = *result.data.at(3);
    assert(data_3.mag == 407580, 'result[3] == 407580'); // 0.04858729
    assert(data_3.sign == false, 'result[3].sign == false');
}

