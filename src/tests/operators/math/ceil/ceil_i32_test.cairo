use array::SpanTrait;
use array::ArrayTrait;
use core::traits::Into;

use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::tests::helpers::tensor::i32::i32_tensor_1x3_helper;
use orion::tests::helpers::tensor::i32::i32_tensor_2x2_helper;
use orion::tests::helpers::tensor::i32::i32_tensor_2x2x2_helper;

#[test]
#[available_gas(2000000000000000000)]
fn tensor_ceil() {
    // ===== 1D ===== //

    let tensor = i32_tensor_1x3_helper();
    let result = tensor.ceil();
    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 1, 'result[1] = 1');
    assert((*result.data[2]).into() == 2, 'result[2] = 2');

    // ===== 2D ===== //

    let tensor = i32_tensor_2x2_helper();
    let result = tensor.ceil();
    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 1, 'result[1] = 1');
    assert((*result.data[2]).into() == 2, 'result[2] = 2');
    assert((*result.data[3]).into() == 3, 'result[3] = 3');

    // ===== 3D ===== //

    let tensor = i32_tensor_2x2x2_helper();
    let result = tensor.ceil();
    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 1, 'result[1] = 1');
    assert((*result.data[2]).into() == 2, 'result[2] = 2');
    assert((*result.data[3]).into() == 3, 'result[3] = 3');
    assert((*result.data[4]).into() == 4, 'result[4] = 4');
    assert((*result.data[5]).into() == 5, 'result[5] = 5');
    assert((*result.data[6]).into() == 6, 'result[6] = 6');
    assert((*result.data[7]).into() == 7, 'result[7] = 7');
}
