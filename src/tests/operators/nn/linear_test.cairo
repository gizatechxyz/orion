use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::nn::core::NNTrait;
use onnx_cairo::operators::nn::implementations::impl_nn_i32;

use debug::PrintTrait;

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
    let inputs = TensorTrait::new(shape.span(), data.span());

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
    let weights = TensorTrait::new(shape.span(), data.span());

    // SET BIAS 
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 61_u32, sign: false });
    data.append(i32 { mag: 71_u32, sign: true });
    let bias = TensorTrait::new(shape.span(), data.span());

    // TEST UNQUANTIZED
    let result = NNTrait::linear(inputs, weights, bias, false).data;
    assert(
        *result.at(0_usize).mag == 5541_u32 & *result.at(0_usize).sign == false, 'result[0] = 5541'
    );
    assert(
        *result.at(1_usize).mag == 260_u32 & *result.at(1_usize).sign == true, 'result[1] = -260'
    );

    // TEST QUANTIZED
    let result = NNTrait::linear(inputs, weights, bias, true).data;
    assert(
        *result.at(0_usize).mag == 127_u32 & *result.at(0_usize).sign == false, 'result[0] = 127'
    );
    assert(*result.at(1_usize).mag == 6_u32 & *result.at(1_usize).sign == true, 'result[1] = -6');
}
