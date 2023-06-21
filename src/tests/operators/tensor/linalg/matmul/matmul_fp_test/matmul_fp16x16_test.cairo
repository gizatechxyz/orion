use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::TensorTrait;
use orion::tests::helpers::tensor::fixed_point::fp16x16::{
    fp_tensor_1x3_helper, fp_tensor_2x2_helper, fp_tensor_3x3_helper
};

#[test]
#[available_gas(200000000)]
fn matmul_1Dx1D() {
    //! Case: Dot product (1D x 1D)
    let tensor_1 = fp_tensor_1x3_helper();
    let tensor_2 = fp_tensor_1x3_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert((*result.data[0]).into() == FixedTrait::from_unscaled_felt(5), 'result[0] = 5');
    assert(result.data.len() == 1, 'data len is 1');
    assert(result.shape.len() == 1, 'shape len is 1');
}

#[test]
#[available_gas(200000000)]
fn matmul_2Dx2D() {
    //! Case: Matrix multiplication (2D x 2D)
    let tensor_1 = fp_tensor_2x2_helper();
    let tensor_2 = fp_tensor_2x2_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert((*result.data[0]).into() == FixedTrait::from_unscaled_felt(2), 'result[0] = 2');
    assert((*result.data[1]).into() == FixedTrait::from_unscaled_felt(3), 'result[1] = 3');
    assert((*result.data[2]).into() == FixedTrait::from_unscaled_felt(6), 'result[2] = 6');
    assert((*result.data[3]).into() == FixedTrait::from_unscaled_felt(11), 'result[3] = 11');
    assert(result.data.len() == 4, 'data len is 4');
    assert(result.shape.len() == 2, 'shape len is 2');
}

#[test]
#[available_gas(200000000)]
fn matmul_2Dx1D() {
    //! Case: Matrix-Vector multiplication (2D x 1D)
    let tensor_1 = fp_tensor_3x3_helper();
    let tensor_2 = fp_tensor_1x3_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert((*result.data[0]).into() == FixedTrait::from_unscaled_felt(5), 'result[0] = 5');
    assert((*result.data[1]).into() == FixedTrait::from_unscaled_felt(14), 'result[1] = 14');
    assert((*result.data[2]).into() == FixedTrait::from_unscaled_felt(23), 'result[2] = 23');
    assert(result.data.len() == 3, 'data len is 3');
    assert(result.shape.len() == 1, 'shape len is 1');
}

#[test]
#[available_gas(200000000)]
fn matmul_1Dx2D() {
    //! Case: Matrix-Vector multiplication (1D x 2D)
    let tensor_1 = fp_tensor_1x3_helper();
    let tensor_2 = fp_tensor_3x3_helper();

    let result = tensor_1.matmul(@tensor_2);
    assert((*result.data[0]).into() == FixedTrait::from_unscaled_felt(15), 'result[0] = 15');
    assert((*result.data[1]).into() == FixedTrait::from_unscaled_felt(18), 'result[1] = 18');
    assert((*result.data[2]).into() == FixedTrait::from_unscaled_felt(21), 'result[2] = 21');
    assert(result.data.len() == 3, 'data len is 3');
    assert(result.shape.len() == 1, 'shape len is 1');
}

