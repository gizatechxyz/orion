use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::{NNTrait};
use orion::operators::nn::implementations::impl_nn_i32;
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::numbers::fixed_point::implementations::impl_8x23;

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
    let alpha = FixedTrait::new(838861, false); // 0.1
    let threshold = IntegerTrait::new(0, false);

    let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

    let data_0 = *result.data.at(0);
    assert(data_0.mag == impl_8x23::ONE, 'result[0] == 8388608'); // 1
    assert(data_0.sign == false, 'result[0].sign == false');

    let data_3 = *result.data.at(3);
    assert(data_3.mag == 1677722, 'result[3] == 1677722'); // 2 * 0.1 = 0.2
    assert(data_3.sign == true, 'result[3].sign == true');

    let data_5 = *result.data.at(5);
    assert(data_5.mag == 0, 'result[5] == 0');
    assert(data_5.sign == false, 'result[5].sign == false');
}

