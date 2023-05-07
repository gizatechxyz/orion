use array::SpanTrait;

use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::tests::operators::tensor::helpers::i32_tensor_2x2x2_helper;

#[test]
#[available_gas(20000000)]
fn tensor_reduce_sum() {
    let tensor = i32_tensor_2x2x2_helper();

    let result = tensor.reduce_sum(0, false);

    assert(*result.data.at(0).mag == 4_u32, 'result[0] = 4');
    assert(*result.data.at(1).mag == 6_u32, 'result[1] = 6');
    assert(*result.data.at(2).mag == 8_u32, 'result[2] = 8');
    assert(*result.data.at(3).mag == 10_u32, 'result[3] = 10');

    let result = tensor.reduce_sum(1, false).data;

    assert(*result.at(0).mag == 2_u32, 'result[0] = 2');
    assert(*result.at(1).mag == 4_u32, 'result[1] = 4');
    assert(*result.at(2).mag == 10_u32, 'result[2] = 10');
    assert(*result.at(3).mag == 12_u32, 'result[3] = 12');

    let result = tensor.reduce_sum(2, false).data;

    assert(*result.at(0).mag == 1_u32, 'result[0] = 1');
    assert(*result.at(1).mag == 5_u32, 'result[1] = 5');
    assert(*result.at(2).mag == 9_u32, 'result[2] = 9');
    assert(*result.at(3).mag == 13_u32, 'result[3] = 13');
}
