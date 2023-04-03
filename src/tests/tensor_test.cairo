use array::ArrayTrait;
use traits::Into;
use debug::print_felt252;

use onnx_cairo::operators::math::int33::i33;
use onnx_cairo::operators::math::tensor::tensor_i33;
use onnx_cairo::operators::math::tensor::core::TensorTrait;
use onnx_cairo::operators::math::tensor::core::Tensor;
use onnx_cairo::operators::math::tensor::core::ravel_index;
use onnx_cairo::operators::math::tensor::core::unravel_index;
use onnx_cairo::operators::math::tensor::tensor_i33::i33_reduce_sum;

#[test]
#[available_gas(2000000)]
#[should_panic]
fn i33_wrong_shape_tensor_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);

    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 0_u32, sign: false });
    data.append(i33 { inner: 1_u32, sign: false });
    data.append(i33 { inner: 2_u32, sign: false });

    let tensor = TensorTrait::<i33>::new(@sizes, @data);
}

#[test]
#[available_gas(2000000)]
fn i33_at_tensor_test() {
    let tensor = i33_tensor_helper();

    let mut indices = ArrayTrait::new();
    indices.append(0_usize);
    indices.append(1_usize);
    indices.append(1_usize);

    let result = tensor.at(@indices).inner;

    assert(result == 3_u32, 'result[3] = 3');
}

#[test]
#[available_gas(2000000)]
fn stride_test() {
    let tensor = i33_tensor_helper();
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
    let result = ravel_index(@shape, @indices);
    assert(result == 2_usize, 'result = 2');

    // 2D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    let mut indices = ArrayTrait::new();
    indices.append(1_usize);
    indices.append(2_usize);
    let result = ravel_index(@shape, @indices);
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
    let result = ravel_index(@shape, @indices);
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
    let result = ravel_index(@shape, @indices);
    assert(result == 142_usize, 'result = 142');
}

#[test]
#[available_gas(2000000)]
fn unravel_index_test() {
    // 1D
    let mut shape = ArrayTrait::new();
    shape.append(5_usize);
    let result = unravel_index(2_usize, @shape);
    assert(*result.at(0_usize) == 2_usize, 'result[0] = 2');

    // 2D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    let result = unravel_index(6_usize, @shape);
    assert(*result.at(0_usize) == 1_usize, 'result[0] = 1');
    assert(*result.at(1_usize) == 2_usize, 'result[1] = 2');

    // 3D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    shape.append(6_usize);
    let result = unravel_index(42_usize, @shape);
    assert(*result.at(0_usize) == 1_usize, 'result[0] = 1');
    assert(*result.at(1_usize) == 3_usize, 'result[1] = 3');
    assert(*result.at(2_usize) == 0_usize, 'result[2] = 0');

    // 4D
    let mut shape = ArrayTrait::new();
    shape.append(2_usize);
    shape.append(4_usize);
    shape.append(6_usize);
    shape.append(8_usize);
    let result = unravel_index(142_usize, @shape);
    assert(*result.at(0_usize) == 0_usize, 'result[0] = 0');
    assert(*result.at(1_usize) == 2_usize, 'result[1] = 2');
    assert(*result.at(2_usize) == 5_usize, 'result[2] = 5');
    assert(*result.at(3_usize) == 6_usize, 'result[3] = 6');
}

#[test]
#[available_gas(2000000)]
fn i33_min_tensor() {
    let tensor = i33_tensor_helper();

    let result = tensor.min().inner;
    assert(result == 0_u32, 'tensor.min = 0');
}

#[test]
#[available_gas(2000000)]
fn i33_max_tensor() {
    let tensor = i33_tensor_helper();

    let result = tensor.max().inner;
    assert(result == 7_u32, 'tensor.max = 7');
}

#[test]
#[available_gas(20000000)]
fn i33_add_tensor() {
    let tensor_1 = i33_tensor_helper();
    let tensor_2 = i33_tensor_helper();

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0_usize).inner == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).inner == 2_u32, 'result[1] = 2');
    assert(*result.at(2_usize).inner == 4_u32, 'result[2] = 4');
    assert(*result.at(3_usize).inner == 6_u32, 'result[3] = 6');
    assert(*result.at(4_usize).inner == 8_u32, 'result[4] = 8');
    assert(*result.at(5_usize).inner == 10_u32, 'result[5] = 10');
    assert(*result.at(6_usize).inner == 12_u32, 'result[6] = 12');
    assert(*result.at(7_usize).inner == 14_u32, 'result[7] = 14');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 10_u32, sign: false });
    data.append(i33 { inner: 100_u32, sign: false });
    let tensor_2 = TensorTrait::<i33>::new(@sizes, @data);

    let result = (tensor_1 + tensor_2).data;

    assert(*result.at(0_usize).inner == 10_u32, 'result[0] = 10');
    assert(*result.at(1_usize).inner == 101_u32, 'result[1] = 101');
    assert(*result.at(2_usize).inner == 12_u32, 'result[2] = 12');
    assert(*result.at(3_usize).inner == 103_u32, 'result[3] = 103');
    assert(*result.at(4_usize).inner == 14_u32, 'result[4] = 14');
    assert(*result.at(5_usize).inner == 105_u32, 'result[5] = 105');
    assert(*result.at(6_usize).inner == 16_u32, 'result[6] = 16');
    assert(*result.at(7_usize).inner == 107_u32, 'result[7] = 107');
}

#[test]
#[available_gas(20000000)]
fn i33_sub_tensor() {
    let tensor_1 = i33_tensor_helper();
    let tensor_2 = i33_tensor_helper();

    let result = (tensor_1 - tensor_2).data;

    assert(*result.at(0_usize).inner == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).inner == 0_u32, 'result[1] = 0');
    assert(*result.at(2_usize).inner == 0_u32, 'result[2] = 0');
    assert(*result.at(3_usize).inner == 0_u32, 'result[3] = 0');
    assert(*result.at(4_usize).inner == 0_u32, 'result[4] = 0');
    assert(*result.at(5_usize).inner == 0_u32, 'result[5] = 0');
    assert(*result.at(6_usize).inner == 0_u32, 'result[6] = 0');
    assert(*result.at(7_usize).inner == 0_u32, 'result[7] = 0');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 0_u32, sign: false });
    data.append(i33 { inner: 1_u32, sign: false });
    let tensor_2 = TensorTrait::<i33>::new(@sizes, @data);

    let result = (tensor_1 - tensor_2).data;

    assert(*result.at(0_usize).inner == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).inner == 0_u32, 'result[1] = 0');
    assert(*result.at(2_usize).inner == 2_u32, 'result[2] = 2');
    assert(*result.at(3_usize).inner == 2_u32, 'result[3] = 2');
    assert(*result.at(4_usize).inner == 4_u32, 'result[4] = 4');
    assert(*result.at(5_usize).inner == 4_u32, 'result[5] = 4');
    assert(*result.at(6_usize).inner == 6_u32, 'result[6] = 6');
    assert(*result.at(7_usize).inner == 6_u32, 'result[7] = 6');
}

#[test]
#[available_gas(20000000)]
fn i33_mul_tensor() {
    let tensor_1 = i33_tensor_helper();
    let tensor_2 = i33_tensor_helper();

    let result = (tensor_1 * tensor_2).data;

    assert(*result.at(0_usize).inner == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).inner == 1_u32, 'result[1] = 1');
    assert(*result.at(2_usize).inner == 4_u32, 'result[2] = 4');
    assert(*result.at(3_usize).inner == 9_u32, 'result[3] = 9');
    assert(*result.at(4_usize).inner == 16_u32, 'result[4] = 16');
    assert(*result.at(5_usize).inner == 25_u32, 'result[5] = 25');
    assert(*result.at(6_usize).inner == 36_u32, 'result[6] = 36');
    assert(*result.at(7_usize).inner == 49_u32, 'result[7] = 49');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 10_u32, sign: false });
    data.append(i33 { inner: 100_u32, sign: false });
    let tensor_2 = TensorTrait::<i33>::new(@sizes, @data);

    let result = (tensor_1 * tensor_2).data;

    assert(*result.at(0_usize).inner == 0_u32, 'result[0] = 0');
    assert(*result.at(1_usize).inner == 100_u32, 'result[1] = 100');
    assert(*result.at(2_usize).inner == 20_u32, 'result[2] = 20');
    assert(*result.at(3_usize).inner == 300_u32, 'result[3] = 300');
    assert(*result.at(4_usize).inner == 40_u32, 'result[4] = 40');
    assert(*result.at(5_usize).inner == 500_u32, 'result[5] = 500');
    assert(*result.at(6_usize).inner == 60_u32, 'result[6] = 60');
    assert(*result.at(7_usize).inner == 700_u32, 'result[7] = 700');
}

#[test]
#[available_gas(20000000)]
fn i33_div_tensor() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);
    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 100_u32, sign: false });
    data.append(i33 { inner: 200_u32, sign: false });
    data.append(i33 { inner: 300_u32, sign: false });
    data.append(i33 { inner: 400_u32, sign: false });
    data.append(i33 { inner: 500_u32, sign: false });
    data.append(i33 { inner: 600_u32, sign: false });
    data.append(i33 { inner: 700_u32, sign: false });
    data.append(i33 { inner: 800_u32, sign: false });
    let tensor_1 = TensorTrait::<i33>::new(@sizes, @data);

    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);
    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 100_u32, sign: false });
    data.append(i33 { inner: 200_u32, sign: false });
    data.append(i33 { inner: 300_u32, sign: false });
    data.append(i33 { inner: 400_u32, sign: false });
    data.append(i33 { inner: 500_u32, sign: false });
    data.append(i33 { inner: 600_u32, sign: false });
    data.append(i33 { inner: 700_u32, sign: false });
    data.append(i33 { inner: 800_u32, sign: false });
    let tensor_2 = TensorTrait::<i33>::new(@sizes, @data);

    let result = (tensor_1 / tensor_2).data;

    assert(*result.at(0_usize).inner == 1_u32, 'result[0] = 1');
    assert(*result.at(1_usize).inner == 1_u32, 'result[1] = 1');
    assert(*result.at(2_usize).inner == 1_u32, 'result[2] = 1');
    assert(*result.at(3_usize).inner == 1_u32, 'result[3] = 1');
    assert(*result.at(4_usize).inner == 1_u32, 'result[4] = 1');
    assert(*result.at(5_usize).inner == 1_u32, 'result[5] = 1');
    assert(*result.at(6_usize).inner == 1_u32, 'result[6] = 1');
    assert(*result.at(7_usize).inner == 1_u32, 'result[7] = 1');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1_usize);
    sizes.append(2_usize);
    sizes.append(1_usize);
    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 10_u32, sign: false });
    data.append(i33 { inner: 100_u32, sign: false });
    let tensor_2 = TensorTrait::<i33>::new(@sizes, @data);

    let result = (tensor_1 / tensor_2).data;

    assert(*result.at(0_usize).inner == 10_u32, 'result[0] = 10');
    assert(*result.at(1_usize).inner == 2_u32, 'result[1] = 2');
    assert(*result.at(2_usize).inner == 30_u32, 'result[2] = 30');
    assert(*result.at(3_usize).inner == 4_u32, 'result[3] = 4');
    assert(*result.at(4_usize).inner == 50_u32, 'result[4] = 50');
    assert(*result.at(5_usize).inner == 6_u32, 'result[5] = 6');
    assert(*result.at(6_usize).inner == 70_u32, 'result[6] = 70');
    assert(*result.at(7_usize).inner == 8_u32, 'result[7] = 8');
}

#[test]
#[available_gas(20000000)]
fn i33_tensor_reduce_sum() {
    let tensor = i33_tensor_helper();

    let result = i33_reduce_sum(@tensor, 0_usize);

    assert(*result.data.at(0_usize).inner == 4_u32, 'result[0] = 4');
    assert(*result.data.at(1_usize).inner == 6_u32, 'result[1] = 6');
    assert(*result.data.at(2_usize).inner == 8_u32, 'result[2] = 8');
    assert(*result.data.at(3_usize).inner == 10_u32, 'result[3] = 10');

    let result = i33_reduce_sum(@tensor, 1_usize);

    assert(*result.data.at(0_usize).inner == 2_u32, 'result[0] = 2');
    assert(*result.data.at(1_usize).inner == 4_u32, 'result[1] = 4');
    assert(*result.data.at(2_usize).inner == 10_u32, 'result[2] = 10');
    assert(*result.data.at(3_usize).inner == 12_u32, 'result[3] = 12');

    let result = i33_reduce_sum(@tensor, 2_usize);

    assert(*result.data.at(0_usize).inner == 1_u32, 'result[0] = 1');
    assert(*result.data.at(1_usize).inner == 5_u32, 'result[1] = 5');
    assert(*result.data.at(2_usize).inner == 9_u32, 'result[2] = 9');
    assert(*result.data.at(3_usize).inner == 13_u32, 'result[3] = 13');
}

fn i33_tensor_helper() -> Tensor<i33> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2_usize);
    sizes.append(2_usize);
    sizes.append(2_usize);

    let mut data = ArrayTrait::new();
    data.append(i33 { inner: 0_u32, sign: false });
    data.append(i33 { inner: 1_u32, sign: false });
    data.append(i33 { inner: 2_u32, sign: false });
    data.append(i33 { inner: 3_u32, sign: false });
    data.append(i33 { inner: 4_u32, sign: false });
    data.append(i33 { inner: 5_u32, sign: false });
    data.append(i33 { inner: 6_u32, sign: false });
    data.append(i33 { inner: 7_u32, sign: false });

    let tensor = TensorTrait::<i33>::new(@sizes, @data);

    return tensor;
}
