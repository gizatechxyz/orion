use array::SpanTrait;
use traits::Into;
use array::ArrayTrait;

use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;

use orion::tests::operators::tensor::helpers::helpers_i32::i32_tensor_2x2_helper;
use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

#[test]
#[available_gas(20000000)]
fn ln_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(4);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);


    let result = tensor.ln().data;

    assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
    assert((*result.at(1).mag).into() == 45355, 'result[1] = 0.69315');
    assert((*result.at(2).mag).into() == 71992, 'result[2] = 1.0986');
    assert((*result.at(3).mag).into() == 301793, 'result[3] = 4.60517');
}


