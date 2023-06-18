use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32;


#[test]
#[available_gas(20000000)]
fn linear_test() {
    // SET INPUTS
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 71_u32, sign: true });
    data.append(i32 { mag: 38_u32, sign: false });
    data.append(i32 { mag: 62_u32, sign: false });
    let extra = Option::<ExtraParams>::None(());
    let inputs = TensorTrait::new(shape.span(), data.span(), extra);

    // SET WEIGHTS
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 8_u32, sign: true });
    data.append(i32 { mag: 64_u32, sign: false });
    data.append(i32 { mag: 40_u32, sign: false });
    data.append(i32 { mag: 33_u32, sign: true });
    data.append(i32 { mag: 34_u32, sign: true });
    data.append(i32 { mag: 20_u32, sign: true });
    let extra = Option::<ExtraParams>::None(());
    let weights = TensorTrait::new(shape.span(), data.span(), extra);

    // SET BIAS 
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 61_u32, sign: false });
    data.append(i32 { mag: 71_u32, sign: true });
    let extra = Option::<ExtraParams>::None(());
    let bias = TensorTrait::new(shape.span(), data.span(), extra);

    // TEST UNQUANTIZED
    let result = NNTrait::linear(inputs, weights, bias, false).data;
    assert((*result[0]).into() == 5541, 'result[0] = 5541');
    assert((*result[1]).into() == -260, 'result[1] = -260');

    // TEST QUANTIZED
    let result = NNTrait::linear(inputs, weights, bias, true).data;
    assert((*result[0]).into() == 127, 'result[0] = 127');
    assert((*result[1]).into() == -6, 'result[1] = -6');
}
