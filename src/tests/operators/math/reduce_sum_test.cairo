use array::SpanTrait;

use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::TensorTrait;
use orion::tests::operators::tensor::helpers::helpers_i32::{i32_tensor_2x2x2_helper, i32_tensor_1x3_helper};

use debug::PrintTrait;

#[test]
#[available_gas(20000000)]
fn reduce_sum_1d() {
    let tensor = i32_tensor_1x3_helper();

    let result = tensor.reduce_sum(0, false);
    assert(*result.data.at(0).mag == 3, 'result[0] = 3');
    assert(result.data.len() == 1, 'result.len = 1');
}

#[test]
#[available_gas(20000000)]
fn reduce_sum_3d() {
    let tensor = i32_tensor_2x2x2_helper();

    let result = tensor.reduce_sum(0, false);

    assert(*result.data.at(0).mag == 4, 'result[0] = 4');
    assert(*result.data.at(1).mag == 6, 'result[1] = 6');
    assert(*result.data.at(2).mag == 8, 'result[2] = 8');
    assert(*result.data.at(3).mag == 10, 'result[3] = 10');

    let result = tensor.reduce_sum(1, false).data;

    assert(*result.at(0).mag == 2, 'result[0] = 2');
    assert(*result.at(1).mag == 4, 'result[1] = 4');
    assert(*result.at(2).mag == 10, 'result[2] = 10');
    assert(*result.at(3).mag == 12, 'result[3] = 12');

    let result = tensor.reduce_sum(2, false).data;

    assert(*result.at(0).mag == 1, 'result[0] = 1');
    assert(*result.at(1).mag == 5, 'result[1] = 5');
    assert(*result.at(2).mag == 9, 'result[2] = 9');
    assert(*result.at(3).mag == 13, 'result[3] = 13');
}
