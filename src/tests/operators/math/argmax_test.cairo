use array::SpanTrait;

use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::tests::operators::tensor::helpers::{i32_tensor_2x2_helper, i32_tensor_2x2x2_helper};

#[test]
#[available_gas(20000000)]
fn tensor_argmax() {
    let tensor = i32_tensor_2x2_helper();

    let result = tensor.argmax(0);
    assert(*result.data.at(0) == 1, 'result[0] = 1');
    assert(*result.data.at(1) == 1, 'result[1] = 1');
    assert(result.data.len() == 2, 'length == 2');

    let result = tensor.argmax(1);

    assert(*result.data.at(0) == 1, 'result[0] = 1');
    assert(*result.data.at(1) == 1, 'result[1] = 1');
    assert(result.data.len() == 2, 'length == 2');

    let tensor = i32_tensor_2x2x2_helper();

    let result = tensor.argmax(0).data;

    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 1, 'result[1] = 1');
    assert(*result.at(2) == 1, 'result[2] = 1');
    assert(*result.at(3) == 1, 'result[3] = 1');
    assert(result.len() == 4, 'length == 4');

    let result = tensor.argmax(1).data;

    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 1, 'result[1] = 1');
    assert(*result.at(2) == 1, 'result[2] = 1');
    assert(*result.at(3) == 1, 'result[3] = 1');
    assert(result.len() == 4, 'length == 4');

    let result = tensor.argmax(2).data;

    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 1, 'result[1] = 1');
    assert(*result.at(2) == 1, 'result[2] = 1');
    assert(*result.at(3) == 1, 'result[3] = 1');
    assert(result.len() == 4, 'length == 4');
}
