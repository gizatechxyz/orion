use array::SpanTrait;
use traits::Into;

use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, };
use orion::tests::operators::tensor::helpers::i32_tensor_2x2_helper;
use orion::numbers::fixed_point::implementations::impl_16x16;

#[test]
#[available_gas(20000000)]
fn tensor_exp_test() {
    let tensor = i32_tensor_2x2_helper();
    let result = tensor.exp().data;

    assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
    assert((*result.at(1).mag).into() == 178142, 'result[1] = 2.7182...');
    assert((*result.at(2).mag).into() == 484232, 'result[2] = 7.389...');
    assert((*result.at(3).mag).into() == 1316288, 'result[3] = 20.085...');
}
