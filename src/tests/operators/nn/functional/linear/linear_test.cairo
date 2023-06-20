use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;


#[test]
#[available_gas(20000000)]
fn linear_test() {
    // SET INPUTS
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 71, sign: true });
    data.append(i32 { mag: 38, sign: false });
    data.append(i32 { mag: 62, sign: false });
    let extra = Option::<ExtraParams>::None(());
    let inputs = TensorTrait::new(shape.span(), data.span(), extra);

    // SET WEIGHTS
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 8, sign: true });
    data.append(i32 { mag: 64, sign: false });
    data.append(i32 { mag: 40, sign: false });
    data.append(i32 { mag: 33, sign: true });
    data.append(i32 { mag: 34, sign: true });
    data.append(i32 { mag: 20, sign: true });
    let extra = Option::<ExtraParams>::None(());
    let weights = TensorTrait::new(shape.span(), data.span(), extra);

    // SET BIAS 
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 61, sign: false });
    data.append(i32 { mag: 71, sign: true });
    let extra = Option::<ExtraParams>::None(());
    let bias = TensorTrait::new(shape.span(), data.span(), extra);

    // TEST UNQUANTIZED
    let result = NNTrait::linear(inputs, weights, bias, false).data;
    assert(
        *result.at(0_usize).mag == 5541 && *result.at(0_usize).sign == false,
        'result[0] = 5541'
    );
    assert(
        *result.at(1_usize).mag == 260 && *result.at(1_usize).sign == true,
        'result[1] = -260'
    );

    // TEST QUANTIZED
    let result = NNTrait::linear(inputs, weights, bias, true).data;
    assert(
        *result.at(0_usize).mag == 127 && *result.at(0_usize).sign == false,
        'result[0] = 127'
    );
    assert(
        *result.at(1_usize).mag == 6 && *result.at(1_usize).sign == true, 'result[1] = -6'
    );
}
