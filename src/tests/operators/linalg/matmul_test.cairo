use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::tests::operators::tensor::helpers::{
    i32_tensor_1x3_helper, i32_tensor_2x2_helper, i32_tensor_3x3_helper
};

#[test]
#[available_gas(200000000)]
fn tensor_matmul() {
    //! Case: Dot product (1D x 1D)
    let tensor_1 = i32_tensor_1x3_helper();
    let tensor_2 = i32_tensor_1x3_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert(*result.data.at(0).mag == 5, 'result[0] = 5');
    assert(result.data.len() == 1, 'data len is 1');
    assert(result.shape.len() == 1, 'shape len is 1');

    //! Case: Matrix multiplication (2D x 2D)
    let tensor_1 = i32_tensor_2x2_helper();
    let tensor_2 = i32_tensor_2x2_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert(*result.data.at(0).mag == 2, 'result[0] = 2');
    assert(*result.data.at(1).mag == 3, 'result[1] = 3');
    assert(*result.data.at(2).mag == 6, 'result[2] = 6');
    assert(*result.data.at(3).mag == 11, 'result[3] = 11');
    assert(result.data.len() == 4, 'data len is 4');
    assert(result.shape.len() == 2, 'shape len is 2');
}

#[test]
#[available_gas(200000000)]
fn tensor_matmul_with_matrix_vec() {
    //! Case: Matrix-Vector multiplication (2D x 1D)
    let tensor_1 = i32_tensor_3x3_helper();
    let tensor_2 = i32_tensor_1x3_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert(*result.data.at(0).mag == 5, 'result[0] = 5');
    assert(*result.data.at(1).mag == 14, 'result[1] = 14');
    assert(*result.data.at(2).mag == 23, 'result[2] = 23');
    assert(result.data.len() == 3, 'data len is 3');
    assert(result.shape.len() == 1, 'shape len is 1');

    //! Case: Matrix-Vector multiplication (1D x 2D)
    let tensor_1 = i32_tensor_1x3_helper();
    let tensor_2 = i32_tensor_3x3_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert(*result.data.at(0).mag == 15, 'result[0] = 15');
    assert(*result.data.at(1).mag == 18, 'result[1] = 18');
    assert(*result.data.at(2).mag == 21, 'result[2] = 21');
    assert(result.data.len() == 3, 'data len is 3');
    assert(result.shape.len() == 1, 'shape len is 1');
}
