use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::operators::math::tensor::tensor_i32;
use onnx_cairo::operators::math::tensor::core::TensorTrait;
use onnx_cairo::operators::math::tensor::core::Tensor;
use onnx_cairo::operators::math::tensor::core::ravel_index;
use onnx_cairo::operators::math::tensor::core::unravel_index;

#[test]
#[available_gas(2000000)]
#[should_panic]
fn wrong_shape_tensor_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), @data);
}

#[test]
#[available_gas(2000000)]
fn at_tensor_test() {
    let tensor = i32_tensor_3d_helper();

    let mut indices = ArrayTrait::new();
    indices.append(0_usize);
    indices.append(1_usize);
    indices.append(1_usize);

    let result = tensor.at(indices.span()).mag;

    assert(result == 3_u32, 'result[3] = 3');
}

#[test]
#[available_gas(2000000)]
fn stride_test() {
    let tensor = i32_tensor_3d_helper();
    let result = tensor.stride();
    assert(*result.at(0_usize) == 4_usize, 'stride x = 4');
    assert(*result.at(1_usize) == 2_usize, 'stride y = 2');
    assert(*result.at(2_usize) == 1_usize, 'stride z = 1');
}

#[test]
#[available_gas(2000000)]
fn ravel_index_test() {
    // 1D
    let mut shape = ArrayTrait::new();
    shape.append(5_usize);
    let mut indices = ArrayTrait::new();
    indices.append(2_usize);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 2_usize, 'result = 2');

    // 2D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    let mut indices = ArrayTrait::new();
    indices.append(1_usize);
    indices.append(2_usize);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 6_usize, 'result = 6');

    // 3D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    shape.append(6_usize);
    let mut indices = ArrayTrait::new();
    indices.append(1_usize);
    indices.append(3_usize);
    indices.append(0_usize);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 42_usize, 'result = 42');

    // 4D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    shape.append(6_usize);
    shape.append(8_usize);
    let mut indices = ArrayTrait::new();
    indices.append(0_usize);
    indices.append(2_usize);
    indices.append(5_usize);
    indices.append(6_usize);
    let result = ravel_index(shape.span(), indices.span());
    assert(result == 142_usize, 'result = 142');
}

#[test]
#[available_gas(2000000)]
fn unravel_index_test() {
    // 1D
    let mut shape = ArrayTrait::new();
    shape.append(5_usize);
    let result = unravel_index(2_usize, shape.span());
    assert(*result.at(0_usize) == 2_usize, 'result[0] = 2');

    // 2D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    let result = unravel_index(6_usize, shape.span());
    assert(*result.at(0_usize) == 1_usize, 'result[0] = 1');
    assert(*result.at(1_usize) == 2_usize, 'result[1] = 2');

    // 3D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    shape.append(6_usize);
    let result = unravel_index(42_usize, shape.span());
    assert(*result.at(0_usize) == 1_usize, 'result[0] = 1');
    assert(*result.at(1_usize) == 3_usize, 'result[1] = 3');
    assert(*result.at(2_usize) == 0_usize, 'result[2] = 0');

    // 4D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    shape.append(6_usize);
    shape.append(8_usize);
    let result = unravel_index(142_usize, shape.span());
    assert(*result.at(0_usize) == 0_usize, 'result[0] = 0');
    assert(*result.at(1_usize) == 2_usize, 'result[1] = 2');
    assert(*result.at(2_usize) == 5_usize, 'result[2] = 5');
    assert(*result.at(3_usize) == 6_usize, 'result[3] = 6');
}

#[test]
#[available_gas(2000000)]
fn min_tensor() {
    let tensor = i32_tensor_3d_helper();

    let result = tensor.min().mag;
    assert(result == 0_u32, 'tensor.min = 0');
}

#[test]
#[available_gas(2000000)]
fn max_tensor() {
    let tensor = i32_tensor_3d_helper();

    let result = tensor.max().mag;
    assert(result == 7_u32, 'tensor.max = 7');
}

#[test]
#[available_gas(20000000)]
fn add_tensor() {
    let tensor_1 = i32_tensor_3d_helper();
    let tensor_2 = i32_tensor_3d_helper();

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0_usize).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).mag == 2_u32, 'result[1] = 2');
    assert(*result.at(2_usize).mag == 4_u32, 'result[2] = 4');
    assert(*result.at(3_usize).mag == 6_u32, 'result[3] = 6');
    assert(*result.at(4_usize).mag == 8_u32, 'result[4] = 8');
    assert(*result.at(5_usize).mag == 10_u32, 'result[5] = 10');
    assert(*result.at(6_usize).mag == 12_u32, 'result[6] = 12');
    assert(*result.at(7_usize).mag == 14_u32, 'result[7] = 14');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), @data);

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0_usize).mag == 10_u32, 'result[0] = 10');
    assert(*result.at(1_usize).mag == 101_u32, 'result[1] = 101');
    assert(*result.at(2_usize).mag == 12_u32, 'result[2] = 12');
    assert(*result.at(3_usize).mag == 103_u32, 'result[3] = 103');
    assert(*result.at(4_usize).mag == 14_u32, 'result[4] = 14');
    assert(*result.at(5_usize).mag == 105_u32, 'result[5] = 105');
    assert(*result.at(6_usize).mag == 16_u32, 'result[6] = 16');
    assert(*result.at(7_usize).mag == 107_u32, 'result[7] = 107');
}

#[test]
#[available_gas(20000000)]
fn sub_tensor() {
    let tensor_1 = i32_tensor_3d_helper();
    let tensor_2 = i32_tensor_3d_helper();

    let result = (tensor_1 - tensor_2).data;

    assert(*result.at(0_usize).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).mag == 0_u32, 'result[1] = 0');
    assert(*result.at(2_usize).mag == 0_u32, 'result[2] = 0');
    assert(*result.at(3_usize).mag == 0_u32, 'result[3] = 0');
    assert(*result.at(4_usize).mag == 0_u32, 'result[4] = 0');
    assert(*result.at(5_usize).mag == 0_u32, 'result[5] = 0');
    assert(*result.at(6_usize).mag == 0_u32, 'result[6] = 0');
    assert(*result.at(7_usize).mag == 0_u32, 'result[7] = 0');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), @data);

    let result = (tensor_1 - tensor_2).data;

    assert(*result.at(0_usize).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).mag == 0_u32, 'result[1] = 0');
    assert(*result.at(2_usize).mag == 2_u32, 'result[2] = 2');
    assert(*result.at(3_usize).mag == 2_u32, 'result[3] = 2');
    assert(*result.at(4_usize).mag == 4_u32, 'result[4] = 4');
    assert(*result.at(5_usize).mag == 4_u32, 'result[5] = 4');
    assert(*result.at(6_usize).mag == 6_u32, 'result[6] = 6');
    assert(*result.at(7_usize).mag == 6_u32, 'result[7] = 6');
}

#[test]
#[available_gas(20000000)]
fn mul_tensor() {
    let tensor_1 = i32_tensor_3d_helper();
    let tensor_2 = i32_tensor_3d_helper();

    let result = (tensor_1 * tensor_2).data;

    assert(*result.at(0_usize).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).mag == 1_u32, 'result[1] = 1');
    assert(*result.at(2_usize).mag == 4_u32, 'result[2] = 4');
    assert(*result.at(3_usize).mag == 9_u32, 'result[3] = 9');
    assert(*result.at(4_usize).mag == 16_u32, 'result[4] = 16');
    assert(*result.at(5_usize).mag == 25_u32, 'result[5] = 25');
    assert(*result.at(6_usize).mag == 36_u32, 'result[6] = 36');
    assert(*result.at(7_usize).mag == 49_u32, 'result[7] = 49');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), @data);

    let result = (tensor_1 * tensor_2).data;

    assert(*result.at(0_usize).mag == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).mag == 100_u32, 'result[1] = 100');
    assert(*result.at(2_usize).mag == 20_u32, 'result[2] = 20');
    assert(*result.at(3_usize).mag == 300_u32, 'result[3] = 300');
    assert(*result.at(4_usize).mag == 40_u32, 'result[4] = 40');
    assert(*result.at(5_usize).mag == 500_u32, 'result[5] = 500');
    assert(*result.at(6_usize).mag == 60_u32, 'result[6] = 60');
    assert(*result.at(7_usize).mag == 700_u32, 'result[7] = 700');
}

#[test]
#[available_gas(20000000)]
fn div_tensor() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(100_u32, false));
    data.append(IntegerTrait::new(200_u32, false));
    data.append(IntegerTrait::new(300_u32, false));
    data.append(IntegerTrait::new(400_u32, false));
    data.append(IntegerTrait::new(500_u32, false));
    data.append(IntegerTrait::new(600_u32, false));
    data.append(IntegerTrait::new(700_u32, false));
    data.append(IntegerTrait::new(800_u32, false));
    let tensor_1 = TensorTrait::<i32>::new(sizes.span(), @data);

    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(100_u32, false));
    data.append(IntegerTrait::new(200_u32, false));
    data.append(IntegerTrait::new(300_u32, false));
    data.append(IntegerTrait::new(400_u32, false));
    data.append(IntegerTrait::new(500_u32, false));
    data.append(IntegerTrait::new(600_u32, false));
    data.append(IntegerTrait::new(700_u32, false));
    data.append(IntegerTrait::new(800_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), @data);

    let result = (tensor_1 / tensor_2).data;

    assert(*result.at(0_usize).mag == 1_u32, 'result[0] = 1');
    assert(*result.at(1_usize).mag == 1_u32, 'result[1] = 1');
    assert(*result.at(2_usize).mag == 1_u32, 'result[2] = 1');
    assert(*result.at(3_usize).mag == 1_u32, 'result[3] = 1');
    assert(*result.at(4_usize).mag == 1_u32, 'result[4] = 1');
    assert(*result.at(5_usize).mag == 1_u32, 'result[5] = 1');
    assert(*result.at(6_usize).mag == 1_u32, 'result[6] = 1');
    assert(*result.at(7_usize).mag == 1_u32, 'result[7] = 1');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), @data);

    let result = (tensor_1 / tensor_2).data;

    assert(*result.at(0_usize).mag == 10_u32, 'result[0] = 10');
    assert(*result.at(1_usize).mag == 2_u32, 'result[1] = 2');
    assert(*result.at(2_usize).mag == 30_u32, 'result[2] = 30');
    assert(*result.at(3_usize).mag == 4_u32, 'result[3] = 4');
    assert(*result.at(4_usize).mag == 50_u32, 'result[4] = 50');
    assert(*result.at(5_usize).mag == 6_u32, 'result[5] = 6');
    assert(*result.at(6_usize).mag == 70_u32, 'result[6] = 70');
    assert(*result.at(7_usize).mag == 8_u32, 'result[7] = 8');
}

#[test]
#[available_gas(20000000)]
fn tensor_reduce_sum() {
    let tensor = i32_tensor_3d_helper();

    let result = tensor.reduce_sum(0_usize);

    assert(*result.data.at(0_usize).mag == 4_u32, 'result[0] = 4');
    assert(*result.data.at(1_usize).mag == 6_u32, 'result[1] = 6');
    assert(*result.data.at(2_usize).mag == 8_u32, 'result[2] = 8');
    assert(*result.data.at(3_usize).mag == 10_u32, 'result[3] = 10');

    let result = tensor.reduce_sum(1_usize);

    assert(*result.data.at(0_usize).mag == 2_u32, 'result[0] = 2');
    assert(*result.data.at(1_usize).mag == 4_u32, 'result[1] = 4');
    assert(*result.data.at(2_usize).mag == 10_u32, 'result[2] = 10');
    assert(*result.data.at(3_usize).mag == 12_u32, 'result[3] = 12');

    let result = tensor.reduce_sum(2_usize);

    assert(*result.data.at(0_usize).mag == 1_u32, 'result[0] = 1');
    assert(*result.data.at(1_usize).mag == 5_u32, 'result[1] = 5');
    assert(*result.data.at(2_usize).mag == 9_u32, 'result[2] = 9');
    assert(*result.data.at(3_usize).mag == 13_u32, 'result[3] = 13');
}

#[test]
#[available_gas(20000000)]
fn tensor_argmax() {
    let tensor = i32_tensor_2d_helper();

    let result = tensor.argmax(0_usize);
assert(*result.data.at(0_usize) == 1_usize, 'result[0] = 1');
assert(*result.data.at(1_usize) == 1_usize, 'result[1] = 1');
assert(result.data.len() == 2_usize, 'length == 2_usize');

let result = tensor.argmax(1_usize);

assert(*result.data.at(0_usize) == 1_usize, 'result[0] = 1');
assert(*result.data.at(1_usize) == 1_usize, 'result[1] = 1');
assert(result.data.len() == 2_usize, 'length == 2_usize');

let tensor = i32_tensor_3d_helper();

let result = tensor.argmax(0_usize);

assert(*result.data.at(0_usize) == 1_usize, 'result[0] = 1');
assert(*result.data.at(1_usize) == 1_usize, 'result[1] = 1');
assert(*result.data.at(2_usize) == 1_usize, 'result[2] = 1');
assert(*result.data.at(3_usize) == 1_usize, 'result[3] = 1');
assert(result.data.len() == 4_usize, 'length == 4_usize');

let result = tensor.argmax(1_usize);

assert(*result.data.at(0_usize) == 1_usize, 'result[0] = 1');
assert(*result.data.at(1_usize) == 1_usize, 'result[1] = 1');
assert(*result.data.at(2_usize) == 1_usize, 'result[2] = 1');
assert(*result.data.at(3_usize) == 1_usize, 'result[3] = 1');
assert(result.data.len() == 4_usize, 'length == 4_usize');

let result = tensor.argmax(2_usize);

assert(*result.data.at(0_usize) == 1_usize, 'result[0] = 1');
assert(*result.data.at(1_usize) == 1_usize, 'result[1] = 1');
assert(*result.data.at(2_usize) == 1_usize, 'result[2] = 1');
assert(*result.data.at(3_usize) == 1_usize, 'result[3] = 1');
assert(result.data.len() == 4_usize, 'length == 4_usize');
}

fn i32_tensor_3d_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));
    data.append(IntegerTrait::new(4_u32, false));
    data.append(IntegerTrait::new(5_u32, false));
    data.append(IntegerTrait::new(6_u32, false));
    data.append(IntegerTrait::new(7_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), @data);

    return tensor;
}

fn i32_tensor_2d_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), @data);

    return tensor;
}

