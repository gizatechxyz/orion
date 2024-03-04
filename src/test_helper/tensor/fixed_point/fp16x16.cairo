use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};

// 1D
fn fp_tensor_1x3_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_1x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

// 2D
fn fp_tensor_2x2_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![2, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(3, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_2x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![2, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true),
        FixedTrait::new_unscaled(3, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x3_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(3, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(5, false),
        FixedTrait::new_unscaled(6, false),
        FixedTrait::new_unscaled(7, false),
        FixedTrait::new_unscaled(8, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true),
        FixedTrait::new_unscaled(3, true),
        FixedTrait::new_unscaled(4, true),
        FixedTrait::new_unscaled(5, true),
        FixedTrait::new_unscaled(6, true),
        FixedTrait::new_unscaled(7, true),
        FixedTrait::new_unscaled(8, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x2_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(3, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(5, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true),
        FixedTrait::new_unscaled(3, true),
        FixedTrait::new_unscaled(4, true),
        FixedTrait::new_unscaled(5, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x1_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 1];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x1_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 1];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_2x3_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![2, 3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(3, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(5, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_2x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![2, 3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true),
        FixedTrait::new_unscaled(3, true),
        FixedTrait::new_unscaled(4, true),
        FixedTrait::new_unscaled(5, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

// 3D
fn fp_tensor_2x2x2_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![2, 2, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(3, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(5, false),
        FixedTrait::new_unscaled(6, false),
        FixedTrait::new_unscaled(7, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_2x2x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![2, 2, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true),
        FixedTrait::new_unscaled(3, true),
        FixedTrait::new_unscaled(4, true),
        FixedTrait::new_unscaled(5, true),
        FixedTrait::new_unscaled(6, true),
        FixedTrait::new_unscaled(7, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x2x2_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 2, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(3, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(5, false),
        FixedTrait::new_unscaled(6, false),
        FixedTrait::new_unscaled(7, false),
        FixedTrait::new_unscaled(8, false),
        FixedTrait::new_unscaled(9, false),
        FixedTrait::new_unscaled(10, false),
        FixedTrait::new_unscaled(11, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x2x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 2, 2];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true),
        FixedTrait::new_unscaled(3, true),
        FixedTrait::new_unscaled(4, true),
        FixedTrait::new_unscaled(5, true),
        FixedTrait::new_unscaled(6, true),
        FixedTrait::new_unscaled(7, true),
        FixedTrait::new_unscaled(8, true),
        FixedTrait::new_unscaled(9, true),
        FixedTrait::new_unscaled(10, true),
        FixedTrait::new_unscaled(11, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}


fn fp_tensor_3x3x3_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 3, 3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, false),
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(3, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(5, false),
        FixedTrait::new_unscaled(6, false),
        FixedTrait::new_unscaled(7, false),
        FixedTrait::new_unscaled(8, false),
        FixedTrait::new_unscaled(9, false),
        FixedTrait::new_unscaled(10, false),
        FixedTrait::new_unscaled(11, false),
        FixedTrait::new_unscaled(12, false),
        FixedTrait::new_unscaled(13, false),
        FixedTrait::new_unscaled(14, false),
        FixedTrait::new_unscaled(15, false),
        FixedTrait::new_unscaled(16, false),
        FixedTrait::new_unscaled(17, false),
        FixedTrait::new_unscaled(18, false),
        FixedTrait::new_unscaled(19, false),
        FixedTrait::new_unscaled(20, false),
        FixedTrait::new_unscaled(21, false),
        FixedTrait::new_unscaled(22, false),
        FixedTrait::new_unscaled(23, false),
        FixedTrait::new_unscaled(24, false),
        FixedTrait::new_unscaled(25, false),
        FixedTrait::new_unscaled(26, false)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

fn fp_tensor_3x3x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes: Array<u32> = array![3, 3, 3];
    let mut data = array![
        FixedTrait::new_unscaled(0, false),
        FixedTrait::new_unscaled(1, true),
        FixedTrait::new_unscaled(2, true),
        FixedTrait::new_unscaled(3, true),
        FixedTrait::new_unscaled(4, true),
        FixedTrait::new_unscaled(5, true),
        FixedTrait::new_unscaled(6, true),
        FixedTrait::new_unscaled(7, true),
        FixedTrait::new_unscaled(8, true),
        FixedTrait::new_unscaled(9, true),
        FixedTrait::new_unscaled(10, true),
        FixedTrait::new_unscaled(11, true),
        FixedTrait::new_unscaled(12, true),
        FixedTrait::new_unscaled(13, true),
        FixedTrait::new_unscaled(14, true),
        FixedTrait::new_unscaled(15, true),
        FixedTrait::new_unscaled(16, true),
        FixedTrait::new_unscaled(17, true),
        FixedTrait::new_unscaled(18, true),
        FixedTrait::new_unscaled(19, true),
        FixedTrait::new_unscaled(20, true),
        FixedTrait::new_unscaled(21, true),
        FixedTrait::new_unscaled(22, true),
        FixedTrait::new_unscaled(23, true),
        FixedTrait::new_unscaled(24, true),
        FixedTrait::new_unscaled(25, true),
        FixedTrait::new_unscaled(26, true)
    ];

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    tensor
}

