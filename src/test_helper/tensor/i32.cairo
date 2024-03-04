use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;

// 1D
fn i32_tensor_1x3_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3];
    let mut data: Array<i32> = array![0, 1, 2];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_1x3_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3];
    let mut data: Array<i32> = array![0, -1, -2];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

// 2D
fn i32_tensor_2x2_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![2, 2];
    let mut data: Array<i32> = array![0, 1, 2, 3];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_2x2_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![2, 2];
    let mut data: Array<i32> = array![0, -1, -2, -3];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x3_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 3];
    let mut data: Array<i32> = array![0, 1, 2, 3, 4, 5, 6, 7, 8];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x3_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 3];
    let mut data: Array<i32> = array![0, -1, -2, -3, -4, -5, -6, -7, -8];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x2_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 2];
    let mut data: Array<i32> = array![0, 1, 2, 3, 4, 5];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x2_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 2];
    let mut data: Array<i32> = array![0, -1, -2, -3, -4, -5];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x1_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 1];
    let mut data: Array<i32> = array![0, 1, 2];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x1_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 1];
    let mut data: Array<i32> = array![0, -1, -2];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_2x3_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![2, 3];
    let mut data: Array<i32> = array![0, 1, 2, 3, 4, 5];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_2x3_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![2, 3];
    let mut data: Array<i32> = array![0, -1, -2, -3, -4, -5];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

// 3D
fn i32_tensor_2x2x2_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![2, 2, 2];
    let mut data: Array<i32> = array![0, 1, 2, 3, 4, 5, 6, 7];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_2x2x2_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![2, 2, 2];
    let mut data: Array<i32> = array![0, -1, -2, -3, -4, -5, -6, -7];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x2x2_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 2, 2];
    let mut data: Array<i32> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x2x2_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 2, 2];
    let mut data: Array<i32> = array![0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x3x3_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 3, 3];
    let mut data: Array<i32> = array![
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    ];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}

fn i32_tensor_3x3x3_neg_helper() -> Tensor<i32> {
    let mut sizes: Array<u32> = array![3, 3, 3];
    let mut data: Array<i32> = array![
        0,
        -1,
        -2,
        -3,
        -4,
        -5,
        -6,
        -7,
        -8,
        -9,
        -10,
        -11,
        -12,
        -13,
        -14,
        -15,
        -16,
        -17,
        -18,
        -19,
        -20,
        -21,
        -22,
        -23,
        -24,
        -25,
        -26
    ];

    let tensor = TensorTrait::new(sizes.span(), data.span());

    tensor
}
