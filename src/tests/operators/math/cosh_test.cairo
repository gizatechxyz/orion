use array::SpanTrait;
use traits::Into;
use debug::PrintTrait;

use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, };
use orion::tests::operators::tensor::helpers::i32_tensor_1x3_helper;
use orion::numbers::fixed_point::implementations::impl_16x16;

#[test]
#[available_gas(20000000)]
fn cosh_test() {
    
    let tensor = i32_tensor_1x3_helper();
    let result = tensor.cosh().data; 

    assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
    assert((*result.at(1).mag).into() == 101125, 'result[1] = 1.5431');
    assert((*result.at(2).mag).into() == 246550, 'result[2] = 3.7622');
}

