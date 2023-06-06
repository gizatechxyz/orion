use core::option::OptionTrait;
use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32;
use orion::numbers::fixed_point::core::FixedImpl;

#[test]
#[available_gas(2000000)]
fn softsign_i32_test() {
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
    let mut result = NNTrait::softsign(@tensor);

    let data = *result.data.at(0);
    assert(data.mag == 0, 'result[0] == 0'); // 0
    assert(data.sign == false, 'result[0].sign == false');

    let data = *result.data.at(1);
    assert(data.mag == 4194304, 'result[1] == 4194304'); // 0.5
    assert(data.sign == false, 'result[1].sign == false');

    let data = *result.data.at(2);
    assert(data.mag == 5592405, 'result[2] == 5592405'); // -0.67
    assert(data.sign == true, 'result[2].sign == true');

    let data = *result.data.at(3);
    assert(data.mag == 6291456, 'result[3] == 6291456'); // -0.75
    assert(data.sign == true, 'result[3].sign == true');
}

