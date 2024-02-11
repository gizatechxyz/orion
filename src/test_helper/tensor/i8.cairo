use core::array::ArrayTrait;
use core::array::SpanTrait;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;


// 1D
fn i8_tensor_1x3_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());
    return tensor;
}

fn i8_tensor_1x3_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());
    return tensor;
}

// 2D

fn i8_tensor_2x2_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);
    data.append(3_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_2x2_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);
    data.append(-3_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x3_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();

    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);
    data.append(3_i8);
    data.append(4_i8);
    data.append(5_i8);
    data.append(6_i8);
    data.append(7_i8);
    data.append(8_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x3_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();

    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);
    data.append(-3_i8);
    data.append(-4_i8);
    data.append(-5_i8);
    data.append(-6_i8);
    data.append(-7_i8);
    data.append(-8_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x2_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data: Array<i8> = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);
    data.append(3_i8);
    data.append(4_i8);
    data.append(5_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x2_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);
    data.append(-3_i8);
    data.append(-4_i8);
    data.append(-5_i8);
    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x1_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x1_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_2x3_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);
    data.append(3_i8);
    data.append(4_i8);
    data.append(5_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_2x3_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);
    data.append(-3_i8);
    data.append(-4_i8);
    data.append(-5_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

// 3D

fn i8_tensor_2x2x2_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);
    data.append(3_i8);
    data.append(4_i8);
    data.append(5_i8);
    data.append(6_i8);
    data.append(7_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_2x2x2_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);
    data.append(-3_i8);
    data.append(-4_i8);
    data.append(-5_i8);
    data.append(-6_i8);
    data.append(-7_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x2x2_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);
    data.append(3_i8);
    data.append(4_i8);
    data.append(5_i8);
    data.append(6_i8);
    data.append(7_i8);
    data.append(8_i8);
    data.append(9_i8);
    data.append(10_i8);
    data.append(11_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x2x2_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);
    data.append(-3_i8);
    data.append(-4_i8);
    data.append(-5_i8);
    data.append(-6_i8);
    data.append(-7_i8);
    data.append(-8_i8);
    data.append(-9_i8);
    data.append(-10_i8);
    data.append(-11_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x3x3_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(1_i8);
    data.append(2_i8);
    data.append(3_i8);
    data.append(4_i8);
    data.append(5_i8);
    data.append(6_i8);
    data.append(7_i8);
    data.append(8_i8);
    data.append(9_i8);
    data.append(10_i8);
    data.append(11_i8);
    data.append(12_i8);
    data.append(13_i8);
    data.append(14_i8);
    data.append(15_i8);
    data.append(16_i8);
    data.append(17_i8);
    data.append(18_i8);
    data.append(19_i8);
    data.append(20_i8);
    data.append(21_i8);
    data.append(22_i8);
    data.append(23_i8);
    data.append(24_i8);
    data.append(25_i8);
    data.append(26_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}

fn i8_tensor_3x3x3_neg_helper() -> Tensor<i8> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0_i8);
    data.append(-1_i8);
    data.append(-2_i8);
    data.append(-3_i8);
    data.append(-4_i8);
    data.append(-5_i8);
    data.append(-6_i8);
    data.append(-7_i8);
    data.append(-8_i8);
    data.append(-9_i8);
    data.append(-10_i8);
    data.append(-11_i8);
    data.append(-12_i8);
    data.append(-13_i8);
    data.append(-14_i8);
    data.append(-15_i8);
    data.append(-16_i8);
    data.append(-17_i8);
    data.append(-18_i8);
    data.append(-19_i8);
    data.append(-20_i8);
    data.append(-21_i8);
    data.append(-22_i8);
    data.append(-23_i8);
    data.append(-24_i8);
    data.append(-25_i8);
    data.append(-26_i8);

    let tensor = TensorTrait::new(sizes.span(), data.span());

    return tensor;
}
