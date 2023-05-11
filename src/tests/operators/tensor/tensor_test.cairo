use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::operators::tensor::core::{TensorTrait, ravel_index, unravel_index};
use onnx_cairo::tests::operators::tensor::helpers::{
    i32_tensor_2x2_helper, i32_tensor_3x2_helper, i32_tensor_2x3_helper, i32_tensor_2x2x2_helper,
    i32_tensor_3x2x2_helper
};

#[test]
#[available_gas(2000000)]
#[should_panic]
fn wrong_shape_tensor_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());
}

#[test]
#[available_gas(2000000)]
fn at_tensor_test() {
    let tensor = i32_tensor_2x2x2_helper();

    let mut indices = ArrayTrait::new();
    indices.append(0);
    indices.append(1);
    indices.append(1);

    let result = tensor.at(indices.span()).mag;

    assert(result == 3_u32, 'result[3] = 3');
}

#[test]
#[available_gas(2000000)]
fn stride_test() {
    let tensor = i32_tensor_2x2x2_helper();
    let result = tensor.stride();
    assert(*result.at(0) == 4, 'stride x = 4');
    assert(*result.at(1) == 2, 'stride y = 2');
    assert(*result.at(2) == 1, 'stride z = 1');
}

#[test]
#[available_gas(2000000)]
fn ravel_index_test() {
    // 1D
    let mut shape = ArrayTrait::new();
    shape.append(5);
    let mut indices = ArrayTrait::new();
    indices.append(2);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 2, 'result = 2');

    // 2D
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(4);
    let mut indices = ArrayTrait::new();
    indices.append(1);
    indices.append(2);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 6, 'result = 6');

    // 3D
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(4);
    shape.append(6);
    let mut indices = ArrayTrait::new();
    indices.append(1);
    indices.append(3);
    indices.append(0);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 42, 'result = 42');

    // 4D
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(4);
    shape.append(6);
    shape.append(8);
    let mut indices = ArrayTrait::new();
    indices.append(0);
    indices.append(2);
    indices.append(5);
    indices.append(6);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 142, 'result = 142');
}

#[test]
#[available_gas(2000000)]
fn unravel_index_test() {
    // 1D
    let mut shape = ArrayTrait::new();
    shape.append(5);
    let result = unravel_index(2, shape.span());
    assert(*result.at(0) == 2, 'result[0] = 2');

    // 2D
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(4);
    let result = unravel_index(6, shape.span());
    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 2, 'result[1] = 2');

    // 3D
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(4);
    shape.append(6);
    let result = unravel_index(42, shape.span());
    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 3, 'result[1] = 3');
    assert(*result.at(2) == 0, 'result[2] = 0');

    // 4D
    let mut shape = ArrayTrait::new();
    shape.append(2);
    shape.append(4);
    shape.append(6);
    shape.append(8);
    let result = unravel_index(142, shape.span());
    assert(*result.at(0) == 0, 'result[0] = 0');
    assert(*result.at(1) == 2, 'result[1] = 2');
    assert(*result.at(2) == 5, 'result[2] = 5');
    assert(*result.at(3) == 6, 'result[3] = 6');
}

#[test]
#[available_gas(20000000)]
fn add_tensor() {
    let tensor_1 = i32_tensor_2x2x2_helper();
    let tensor_2 = i32_tensor_2x2x2_helper();

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1).mag == 2_u32, 'result[1] = 2');
    assert(*result.at(2).mag == 4_u32, 'result[2] = 4');
    assert(*result.at(3).mag == 6_u32, 'result[3] = 6');
    assert(*result.at(4).mag == 8_u32, 'result[4] = 8');
    assert(*result.at(5).mag == 10_u32, 'result[5] = 10');
    assert(*result.at(6).mag == 12_u32, 'result[6] = 12');
    assert(*result.at(7).mag == 14_u32, 'result[7] = 14');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0).mag == 10_u32, 'result[0] = 10');
    assert(*result.at(1).mag == 11_u32, 'result[1] = 11');
    assert(*result.at(2).mag == 102_u32, 'result[2] = 102');
    assert(*result.at(3).mag == 103_u32, 'result[3] = 103');
    assert(*result.at(4).mag == 14_u32, 'result[4] = 14');
    assert(*result.at(5).mag == 15_u32, 'result[5] = 15');
    assert(*result.at(6).mag == 106_u32, 'result[6] = 106');
    assert(*result.at(7).mag == 107_u32, 'result[7] = 107');

    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(1);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0).mag == 10_u32, 'result[0] = 10');
    assert(*result.at(1).mag == 11_u32, 'result[1] = 11');
    assert(*result.at(2).mag == 12_u32, 'result[2] = 12');
    assert(*result.at(3).mag == 13_u32, 'result[3] = 13');
    assert(*result.at(4).mag == 104_u32, 'result[4] = 104');
    assert(*result.at(5).mag == 105_u32, 'result[5] = 105');
    assert(*result.at(6).mag == 106_u32, 'result[6] = 106');
    assert(*result.at(7).mag == 107_u32, 'result[7] = 107');

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(1);
    sizes.append(2);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0).mag == 10_u32, 'result[0] = 10');
    assert(*result.at(1).mag == 101_u32, 'result[1] = 101');
    assert(*result.at(2).mag == 12_u32, 'result[2] = 12');
    assert(*result.at(3).mag == 103_u32, 'result[3] = 103');
    assert(*result.at(4).mag == 14_u32, 'result[4] = 14');
    assert(*result.at(5).mag == 105_u32, 'result[5] = 105');
    assert(*result.at(6).mag == 16_u32, 'result[6] = 16');
    assert(*result.at(7).mag == 107_u32, 'result[7] = 107');
}

#[test]
#[available_gas(200000000)]
fn sub_tensor() {
    let tensor_1 = i32_tensor_2x2x2_helper();
    let tensor_2 = i32_tensor_2x2x2_helper();

    let result = (tensor_1 - tensor_2).data;

    assert(*result.at(0).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1).mag == 0_u32, 'result[1] = 0');
    assert(*result.at(2).mag == 0_u32, 'result[2] = 0');
    assert(*result.at(3).mag == 0_u32, 'result[3] = 0');
    assert(*result.at(4).mag == 0_u32, 'result[4] = 0');
    assert(*result.at(5).mag == 0_u32, 'result[5] = 0');
    assert(*result.at(6).mag == 0_u32, 'result[6] = 0');
    assert(*result.at(7).mag == 0_u32, 'result[7] = 0');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let result = (tensor_1 - tensor_2).data;

    assert(*result.at(0).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1).mag == 1_u32, 'result[1] = 1');
    assert(*result.at(2).mag == 1_u32, 'result[2] = 1');
    assert(*result.at(3).mag == 2_u32, 'result[3] = 2');
    assert(*result.at(4).mag == 4_u32, 'result[4] = 4');
    assert(*result.at(5).mag == 5_u32, 'result[5] = 5');
    assert(*result.at(6).mag == 5_u32, 'result[6] = 5');
    assert(*result.at(7).mag == 6_u32, 'result[7] = 6');
}

#[test]
#[available_gas(20000000)]
fn mul_tensor() {
    let tensor_1 = i32_tensor_2x2x2_helper();
    let tensor_2 = i32_tensor_2x2x2_helper();

    let result = (tensor_1 * tensor_2).data;

    assert(*result.at(0).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1).mag == 1_u32, 'result[1] = 1');
    assert(*result.at(2).mag == 4_u32, 'result[2] = 4');
    assert(*result.at(3).mag == 9_u32, 'result[3] = 9');
    assert(*result.at(4).mag == 16_u32, 'result[4] = 16');
    assert(*result.at(5).mag == 25_u32, 'result[5] = 25');
    assert(*result.at(6).mag == 36_u32, 'result[6] = 36');
    assert(*result.at(7).mag == 49_u32, 'result[7] = 49');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let result = (tensor_1 * tensor_2).data;

    assert(*result.at(0).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1).mag == 10_u32, 'result[1] = 10');
    assert(*result.at(2).mag == 200_u32, 'result[2] = 200');
    assert(*result.at(3).mag == 300_u32, 'result[3] = 300');
    assert(*result.at(4).mag == 40_u32, 'result[4] = 40');
    assert(*result.at(5).mag == 50_u32, 'result[5] = 50');
    assert(*result.at(6).mag == 600_u32, 'result[6] = 600');
    assert(*result.at(7).mag == 700_u32, 'result[7] = 700');
}

#[test]
#[available_gas(20000000)]
fn div_tensor() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(100_u32, false));
    data.append(IntegerTrait::new(200_u32, false));
    data.append(IntegerTrait::new(300_u32, false));
    data.append(IntegerTrait::new(400_u32, false));
    data.append(IntegerTrait::new(500_u32, false));
    data.append(IntegerTrait::new(600_u32, false));
    data.append(IntegerTrait::new(700_u32, false));
    data.append(IntegerTrait::new(800_u32, false));
    let tensor_1 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(100_u32, false));
    data.append(IntegerTrait::new(200_u32, false));
    data.append(IntegerTrait::new(300_u32, false));
    data.append(IntegerTrait::new(400_u32, false));
    data.append(IntegerTrait::new(500_u32, false));
    data.append(IntegerTrait::new(600_u32, false));
    data.append(IntegerTrait::new(700_u32, false));
    data.append(IntegerTrait::new(800_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let result = (tensor_1 / tensor_2).data;

    assert(*result.at(0).mag == 1_u32, 'result[0] = 1');
    assert(*result.at(1).mag == 1_u32, 'result[1] = 1');
    assert(*result.at(2).mag == 1_u32, 'result[2] = 1');
    assert(*result.at(3).mag == 1_u32, 'result[3] = 1');
    assert(*result.at(4).mag == 1_u32, 'result[4] = 1');
    assert(*result.at(5).mag == 1_u32, 'result[5] = 1');
    assert(*result.at(6).mag == 1_u32, 'result[6] = 1');
    assert(*result.at(7).mag == 1_u32, 'result[7] = 1');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span());

    let result = (tensor_1 / tensor_2).data;

    assert(*result.at(0).mag == 10_u32, 'result[0] = 10');
    assert(*result.at(1).mag == 20_u32, 'result[1] = 20');
    assert(*result.at(2).mag == 3_u32, 'result[2] = 3');
    assert(*result.at(3).mag == 4_u32, 'result[3] = 4');
    assert(*result.at(4).mag == 50_u32, 'result[4] = 50');
    assert(*result.at(5).mag == 60_u32, 'result[5] = 60');
    assert(*result.at(6).mag == 7_u32, 'result[6] = 7');
    assert(*result.at(7).mag == 8_u32, 'result[7] = 8');
}

#[test]
#[available_gas(200000000)]
fn tensor_transpose_2D() {
    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(1);
    axes.append(0);

    let tensor = i32_tensor_2x2_helper();

    let result = tensor.transpose(axes.span());

    assert(*result.data.at(0).mag == 0, 'result[0] = 0');
    assert(*result.data.at(1).mag == 2, 'result[1] = 2');
    assert(*result.data.at(2).mag == 1, 'result[2] = 1');
    assert(*result.data.at(3).mag == 3, 'result[3] = 3');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');

    let tensor = i32_tensor_3x2_helper();

    let result = tensor.transpose(axes.span());

    assert(*result.data.at(0).mag == 0, 'result[0] = 0');
    assert(*result.data.at(1).mag == 2, 'result[1] = 2');
    assert(*result.data.at(2).mag == 4, 'result[2] = 4');
    assert(*result.data.at(3).mag == 1, 'result[3] = 1');
    assert(*result.data.at(4).mag == 3, 'result[4] = 3');
    assert(*result.data.at(5).mag == 5, 'result[5] = 5');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 3, 'shape[1] = 3');

    let tensor = i32_tensor_2x3_helper();

    let result = tensor.transpose(axes.span());

    assert(*result.data.at(0).mag == 0, 'result[0] = 0');
    assert(*result.data.at(1).mag == 3, 'result[1] = 3');
    assert(*result.data.at(2).mag == 1, 'result[2] = 1');
    assert(*result.data.at(3).mag == 4, 'result[3] = 4');
    assert(*result.data.at(4).mag == 2, 'result[4] = 2');
    assert(*result.data.at(5).mag == 5, 'result[5] = 5');
    assert(*result.shape.at(0) == 3, 'shape[0] = 3');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');
}

#[test]
#[available_gas(200000000)]
fn tensor_transpose_3D() {
    let tensor = i32_tensor_2x2x2_helper();

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(1);
    axes.append(2);
    axes.append(0);

    let result = tensor.transpose(axes.span()).data;

    assert(*result.at(0).mag == 0, 'result[0] = 0');
    assert(*result.at(1).mag == 4, 'result[1] = 4');
    assert(*result.at(2).mag == 1, 'result[2] = 1');
    assert(*result.at(3).mag == 5, 'result[3] = 5');
    assert(*result.at(4).mag == 2, 'result[4] = 2');
    assert(*result.at(5).mag == 6, 'result[5] = 6');
    assert(*result.at(6).mag == 3, 'result[6] = 3');
    assert(*result.at(7).mag == 7, 'result[7] = 7');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(2);
    axes.append(1);
    axes.append(0);

    let result = tensor.transpose(axes.span()).data;

    assert(*result.at(0).mag == 0, 'result[0] = 0');
    assert(*result.at(1).mag == 4, 'result[1] = 4');
    assert(*result.at(2).mag == 2, 'result[2] = 2');
    assert(*result.at(3).mag == 6, 'result[3] = 6');
    assert(*result.at(4).mag == 1, 'result[4] = 1');
    assert(*result.at(5).mag == 5, 'result[5] = 5');
    assert(*result.at(6).mag == 3, 'result[6] = 3');
    assert(*result.at(7).mag == 7, 'result[7] = 7');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(0);
    axes.append(2);
    axes.append(1);

    let result = tensor.transpose(axes.span()).data;

    assert(*result.at(0).mag == 0, 'result[0] = 0');
    assert(*result.at(1).mag == 2, 'result[1] = 2');
    assert(*result.at(2).mag == 1, 'result[2] = 1');
    assert(*result.at(3).mag == 3, 'result[3] = 3');
    assert(*result.at(4).mag == 4, 'result[4] = 4');
    assert(*result.at(5).mag == 6, 'result[5] = 6');
    assert(*result.at(6).mag == 5, 'result[6] = 5');
    assert(*result.at(7).mag == 7, 'result[7] = 7');

    let tensor = i32_tensor_3x2x2_helper();

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(1);
    axes.append(2);
    axes.append(0);

    let result = tensor.transpose(axes.span());

    assert(*result.data.at(0).mag == 0, 'result[0] = 0');
    assert(*result.data.at(1).mag == 4, 'result[1] = 4');
    assert(*result.data.at(2).mag == 8, 'result[2] = 8');
    assert(*result.data.at(3).mag == 1, 'result[3] = 1');
    assert(*result.data.at(4).mag == 5, 'result[4] = 5');
    assert(*result.data.at(5).mag == 9, 'result[5] = 9');
    assert(*result.data.at(6).mag == 2, 'result[6] = 2');
    assert(*result.data.at(7).mag == 6, 'result[7] = 6');
    assert(*result.data.at(8).mag == 10, 'result[7] = 10');
    assert(*result.data.at(9).mag == 3, 'result[7] = 3');
    assert(*result.data.at(10).mag == 7, 'result[7] = 7');
    assert(*result.data.at(11).mag == 11, 'result[7] = 11');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');
    assert(*result.shape.at(2) == 3, 'shape[2] = 3');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(2);
    axes.append(1);
    axes.append(0);

    let result = tensor.transpose(axes.span());

    assert(*result.data.at(0).mag == 0, 'result[0] = 0');
    assert(*result.data.at(1).mag == 4, 'result[1] = 4');
    assert(*result.data.at(2).mag == 8, 'result[2] = 8');
    assert(*result.data.at(3).mag == 2, 'result[3] = 2');
    assert(*result.data.at(4).mag == 6, 'result[4] = 6');
    assert(*result.data.at(5).mag == 10, 'result[5] = 10');
    assert(*result.data.at(6).mag == 1, 'result[6] = 1');
    assert(*result.data.at(7).mag == 5, 'result[7] = 5');
    assert(*result.data.at(8).mag == 9, 'result[7] = 9');
    assert(*result.data.at(9).mag == 3, 'result[7] = 3');
    assert(*result.data.at(10).mag == 7, 'result[7] = 7');
    assert(*result.data.at(11).mag == 11, 'result[7] = 11');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');
    assert(*result.shape.at(2) == 3, 'shape[2] = 3');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(0);
    axes.append(2);
    axes.append(1);

    let result = tensor.transpose(axes.span());

    assert(*result.data.at(0).mag == 0, 'result[0] = 0');
    assert(*result.data.at(1).mag == 2, 'result[1] = 2');
    assert(*result.data.at(2).mag == 1, 'result[2] = 1');
    assert(*result.data.at(3).mag == 3, 'result[3] = 3');
    assert(*result.data.at(4).mag == 4, 'result[4] = 4');
    assert(*result.data.at(5).mag == 6, 'result[5] = 6');
    assert(*result.data.at(6).mag == 5, 'result[6] = 5');
    assert(*result.data.at(7).mag == 7, 'result[7] = 7');
    assert(*result.data.at(8).mag == 8, 'result[7] = 8');
    assert(*result.data.at(9).mag == 10, 'result[7] = 10');
    assert(*result.data.at(10).mag == 9, 'result[7] = 9');
    assert(*result.data.at(11).mag == 11, 'result[7] = 11');
    assert(*result.shape.at(0) == 3, 'shape[0] = 3');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');
    assert(*result.shape.at(2) == 2, 'shape[2] = 2');
}
