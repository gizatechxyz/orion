use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, ExtraParams, ravel_index, unravel_index};
use orion::tests::helpers::tensor::i32::{
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

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
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

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 2, 'result[1] = 2');
    assert((*result[2]).into() == 4, 'result[2] = 4');
    assert((*result[3]).into() == 6, 'result[3] = 6');
    assert((*result[4]).into() == 8, 'result[4] = 8');
    assert((*result[5]).into() == 10, 'result[5] = 10');
    assert((*result[6]).into() == 12, 'result[6] = 12');
    assert((*result[7]).into() == 14, 'result[7] = 14');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = (tensor_1 + tensor_2).data;

    assert((*result[0]).into() == 10, 'result[0] = 10');
    assert((*result[1]).into() == 11, 'result[1] = 11');
    assert((*result[2]).into() == 102, 'result[2] = 102');
    assert((*result[3]).into() == 103, 'result[3] = 103');
    assert((*result[4]).into() == 14, 'result[4] = 14');
    assert((*result[5]).into() == 15, 'result[5] = 15');
    assert((*result[6]).into() == 106, 'result[6] = 106');
    assert((*result[7]).into() == 107, 'result[7] = 107');

    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(1);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = (tensor_1 + tensor_2).data;

    assert((*result[0]).into() == 10, 'result[0] = 10');
    assert((*result[1]).into() == 11, 'result[1] = 11');
    assert((*result[2]).into() == 12, 'result[2] = 12');
    assert((*result[3]).into() == 13, 'result[3] = 13');
    assert((*result[4]).into() == 104, 'result[4] = 104');
    assert((*result[5]).into() == 105, 'result[5] = 105');
    assert((*result[6]).into() == 106, 'result[6] = 106');
    assert((*result[7]).into() == 107, 'result[7] = 107');

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(1);
    sizes.append(2);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = (tensor_1 + tensor_2).data;

    assert((*result[0]).into() == 10, 'result[0] = 10');
    assert((*result[1]).into() == 101, 'result[1] = 101');
    assert((*result[2]).into() == 12, 'result[2] = 12');
    assert((*result[3]).into() == 103, 'result[3] = 103');
    assert((*result[4]).into() == 14, 'result[4] = 14');
    assert((*result[5]).into() == 105, 'result[5] = 105');
    assert((*result[6]).into() == 16, 'result[6] = 16');
    assert((*result[7]).into() == 107, 'result[7] = 107');
}

#[test]
#[available_gas(200000000)]
fn sub_tensor() {
    let tensor_1 = i32_tensor_2x2x2_helper();
    let tensor_2 = i32_tensor_2x2x2_helper();

    let result = (tensor_1 - tensor_2).data;

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 0, 'result[1] = 0');
    assert((*result[2]).into() == 0, 'result[2] = 0');
    assert((*result[3]).into() == 0, 'result[3] = 0');
    assert((*result[4]).into() == 0, 'result[4] = 0');
    assert((*result[5]).into() == 0, 'result[5] = 0');
    assert((*result[6]).into() == 0, 'result[6] = 0');
    assert((*result[7]).into() == 0, 'result[7] = 0');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = (tensor_1 - tensor_2).data;

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 1, 'result[1] = 1');
    assert((*result[2]).into() == 1, 'result[2] = 1');
    assert((*result[3]).into() == 2, 'result[3] = 2');
    assert((*result[4]).into() == 4, 'result[4] = 4');
    assert((*result[5]).into() == 5, 'result[5] = 5');
    assert((*result[6]).into() == 5, 'result[6] = 5');
    assert((*result[7]).into() == 6, 'result[7] = 6');
}

#[test]
#[available_gas(20000000)]
fn mul_tensor() {
    let tensor_1 = i32_tensor_2x2x2_helper();
    let tensor_2 = i32_tensor_2x2x2_helper();

    let result = (tensor_1 * tensor_2).data;

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 1, 'result[1] = 1');
    assert((*result[2]).into() == 4, 'result[2] = 4');
    assert((*result[3]).into() == 9, 'result[3] = 9');
    assert((*result[4]).into() == 16, 'result[4] = 16');
    assert((*result[5]).into() == 25, 'result[5] = 25');
    assert((*result[6]).into() == 36, 'result[6] = 36');
    assert((*result[7]).into() == 49, 'result[7] = 49');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = (tensor_1 * tensor_2).data;

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 10, 'result[1] = 10');
    assert((*result[2]).into() == 200, 'result[2] = 200');
    assert((*result[3]).into() == 300, 'result[3] = 300');
    assert((*result[4]).into() == 40, 'result[4] = 40');
    assert((*result[5]).into() == 50, 'result[5] = 50');
    assert((*result[6]).into() == 600, 'result[6] = 600');
    assert((*result[7]).into() == 700, 'result[7] = 700');
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
    let extra = Option::<ExtraParams>::None(());
    let tensor_1 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

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
    let extra = Option::<ExtraParams>::None(());
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = (tensor_1 / tensor_2).data;

    assert((*result[0]).into() == 1, 'result[0] = 1');
    assert((*result[1]).into() == 1, 'result[1] = 1');
    assert((*result[2]).into() == 1, 'result[2] = 1');
    assert((*result[3]).into() == 1, 'result[3] = 1');
    assert((*result[4]).into() == 1, 'result[4] = 1');
    assert((*result[5]).into() == 1, 'result[5] = 1');
    assert((*result[6]).into() == 1, 'result[6] = 1');
    assert((*result[7]).into() == 1, 'result[7] = 1');

    // broadcast operation 

    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(2);
    sizes.append(1);
    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(100_u32, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor_2 = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    let result = (tensor_1 / tensor_2).data;

    assert((*result[0]).into() == 10, 'result[0] = 10');
    assert((*result[1]).into() == 20, 'result[1] = 20');
    assert((*result[2]).into() == 3, 'result[2] = 3');
    assert((*result[3]).into() == 4, 'result[3] = 4');
    assert((*result[4]).into() == 50, 'result[4] = 50');
    assert((*result[5]).into() == 60, 'result[5] = 60');
    assert((*result[6]).into() == 7, 'result[6] = 7');
    assert((*result[7]).into() == 8, 'result[7] = 8');
}

#[test]
#[available_gas(200000000)]
fn tensor_transpose_2D() {
    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(1);
    axes.append(0);

    let tensor = i32_tensor_2x2_helper();

    let result = tensor.transpose(axes.span());

    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 2, 'result[1] = 2');
    assert((*result.data[2]).into() == 1, 'result[2] = 1');
    assert((*result.data[3]).into() == 3, 'result[3] = 3');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');

    let tensor = i32_tensor_3x2_helper();

    let result = tensor.transpose(axes.span());

    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 2, 'result[1] = 2');
    assert((*result.data[2]).into() == 4, 'result[2] = 4');
    assert((*result.data[3]).into() == 1, 'result[3] = 1');
    assert((*result.data[4]).into() == 3, 'result[4] = 3');
    assert((*result.data[5]).into() == 5, 'result[5] = 5');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 3, 'shape[1] = 3');

    let tensor = i32_tensor_2x3_helper();

    let result = tensor.transpose(axes.span());

    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 3, 'result[1] = 3');
    assert((*result.data[2]).into() == 1, 'result[2] = 1');
    assert((*result.data[3]).into() == 4, 'result[3] = 4');
    assert((*result.data[4]).into() == 2, 'result[4] = 2');
    assert((*result.data[5]).into() == 5, 'result[5] = 5');
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

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 4, 'result[1] = 4');
    assert((*result[2]).into() == 1, 'result[2] = 1');
    assert((*result[3]).into() == 5, 'result[3] = 5');
    assert((*result[4]).into() == 2, 'result[4] = 2');
    assert((*result[5]).into() == 6, 'result[5] = 6');
    assert((*result[6]).into() == 3, 'result[6] = 3');
    assert((*result[7]).into() == 7, 'result[7] = 7');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(2);
    axes.append(1);
    axes.append(0);

    let result = tensor.transpose(axes.span()).data;

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 4, 'result[1] = 4');
    assert((*result[2]).into() == 2, 'result[2] = 2');
    assert((*result[3]).into() == 6, 'result[3] = 6');
    assert((*result[4]).into() == 1, 'result[4] = 1');
    assert((*result[5]).into() == 5, 'result[5] = 5');
    assert((*result[6]).into() == 3, 'result[6] = 3');
    assert((*result[7]).into() == 7, 'result[7] = 7');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(0);
    axes.append(2);
    axes.append(1);

    let result = tensor.transpose(axes.span()).data;

    assert((*result[0]).into() == 0, 'result[0] = 0');
    assert((*result[1]).into() == 2, 'result[1] = 2');
    assert((*result[2]).into() == 1, 'result[2] = 1');
    assert((*result[3]).into() == 3, 'result[3] = 3');
    assert((*result[4]).into() == 4, 'result[4] = 4');
    assert((*result[5]).into() == 6, 'result[5] = 6');
    assert((*result[6]).into() == 5, 'result[6] = 5');
    assert((*result[7]).into() == 7, 'result[7] = 7');

    let tensor = i32_tensor_3x2x2_helper();

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(1);
    axes.append(2);
    axes.append(0);

    let result = tensor.transpose(axes.span());

    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 4, 'result[1] = 4');
    assert((*result.data[2]).into() == 8, 'result[2] = 8');
    assert((*result.data[3]).into() == 1, 'result[3] = 1');
    assert((*result.data[4]).into() == 5, 'result[4] = 5');
    assert((*result.data[5]).into() == 9, 'result[5] = 9');
    assert((*result.data[6]).into() == 2, 'result[6] = 2');
    assert((*result.data[7]).into() == 6, 'result[7] = 6');
    assert((*result.data[8]).into() == 10, 'result[7] = 10');
    assert((*result.data[9]).into() == 3, 'result[7] = 3');
    assert((*result.data[10]).into() == 7, 'result[7] = 7');
    assert((*result.data[11]).into() == 11, 'result[7] = 11');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');
    assert(*result.shape.at(2) == 3, 'shape[2] = 3');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(2);
    axes.append(1);
    axes.append(0);

    let result = tensor.transpose(axes.span());

    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 4, 'result[1] = 4');
    assert((*result.data[2]).into() == 8, 'result[2] = 8');
    assert((*result.data[3]).into() == 2, 'result[3] = 2');
    assert((*result.data[4]).into() == 6, 'result[4] = 6');
    assert((*result.data[5]).into() == 10, 'result[5] = 10');
    assert((*result.data[6]).into() == 1, 'result[6] = 1');
    assert((*result.data[7]).into() == 5, 'result[7] = 5');
    assert((*result.data[8]).into() == 9, 'result[7] = 9');
    assert((*result.data[9]).into() == 3, 'result[7] = 3');
    assert((*result.data[10]).into() == 7, 'result[7] = 7');
    assert((*result.data[11]).into() == 11, 'result[7] = 11');
    assert(*result.shape.at(0) == 2, 'shape[0] = 2');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');
    assert(*result.shape.at(2) == 3, 'shape[2] = 3');

    let mut axes: Array<usize> = ArrayTrait::new();
    axes.append(0);
    axes.append(2);
    axes.append(1);

    let result = tensor.transpose(axes.span());

    assert((*result.data[0]).into() == 0, 'result[0] = 0');
    assert((*result.data[1]).into() == 2, 'result[1] = 2');
    assert((*result.data[2]).into() == 1, 'result[2] = 1');
    assert((*result.data[3]).into() == 3, 'result[3] = 3');
    assert((*result.data[4]).into() == 4, 'result[4] = 4');
    assert((*result.data[5]).into() == 6, 'result[5] = 6');
    assert((*result.data[6]).into() == 5, 'result[6] = 5');
    assert((*result.data[7]).into() == 7, 'result[7] = 7');
    assert((*result.data[8]).into() == 8, 'result[7] = 8');
    assert((*result.data[9]).into() == 10, 'result[7] = 10');
    assert((*result.data[10]).into() == 9, 'result[7] = 9');
    assert((*result.data[11]).into() == 11, 'result[7] = 11');
    assert(*result.shape.at(0) == 3, 'shape[0] = 3');
    assert(*result.shape.at(1) == 2, 'shape[1] = 2');
    assert(*result.shape.at(2) == 2, 'shape[2] = 2');
}
