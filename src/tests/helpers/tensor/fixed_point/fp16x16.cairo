use array::ArrayTrait;
use array::SpanTrait;
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};

// 1D
fn fp_tensor_1x3_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_1x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

// 2D

fn fp_tensor_2x2_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_2x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x3_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));
    data.append(FixedTrait::new_unscaled(6, false));
    data.append(FixedTrait::new_unscaled(7, false));
    data.append(FixedTrait::new_unscaled(8, false));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));
    data.append(FixedTrait::new_unscaled(6, true));
    data.append(FixedTrait::new_unscaled(7, true));
    data.append(FixedTrait::new_unscaled(8, true));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x2_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x1_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x1_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_2x3_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_2x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

// 3D

fn fp_tensor_2x2x2_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));
    data.append(FixedTrait::new_unscaled(6, false));
    data.append(FixedTrait::new_unscaled(7, false));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_2x2x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));
    data.append(FixedTrait::new_unscaled(6, true));
    data.append(FixedTrait::new_unscaled(7, true));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x2x2_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));
    data.append(FixedTrait::new_unscaled(6, false));
    data.append(FixedTrait::new_unscaled(7, false));
    data.append(FixedTrait::new_unscaled(8, false));
    data.append(FixedTrait::new_unscaled(9, false));
    data.append(FixedTrait::new_unscaled(10, false));
    data.append(FixedTrait::new_unscaled(11, false));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x2x2_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));
    data.append(FixedTrait::new_unscaled(6, true));
    data.append(FixedTrait::new_unscaled(7, true));
    data.append(FixedTrait::new_unscaled(8, true));
    data.append(FixedTrait::new_unscaled(9, true));
    data.append(FixedTrait::new_unscaled(10, true));
    data.append(FixedTrait::new_unscaled(11, true));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x3x3_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));
    data.append(FixedTrait::new_unscaled(6, false));
    data.append(FixedTrait::new_unscaled(7, false));
    data.append(FixedTrait::new_unscaled(8, false));
    data.append(FixedTrait::new_unscaled(9, false));
    data.append(FixedTrait::new_unscaled(10, false));
    data.append(FixedTrait::new_unscaled(11, false));
    data.append(FixedTrait::new_unscaled(12, false));
    data.append(FixedTrait::new_unscaled(13, false));
    data.append(FixedTrait::new_unscaled(14, false));
    data.append(FixedTrait::new_unscaled(15, false));
    data.append(FixedTrait::new_unscaled(16, false));
    data.append(FixedTrait::new_unscaled(17, false));
    data.append(FixedTrait::new_unscaled(18, false));
    data.append(FixedTrait::new_unscaled(19, false));
    data.append(FixedTrait::new_unscaled(20, false));
    data.append(FixedTrait::new_unscaled(21, false));
    data.append(FixedTrait::new_unscaled(22, false));
    data.append(FixedTrait::new_unscaled(23, false));
    data.append(FixedTrait::new_unscaled(24, false));
    data.append(FixedTrait::new_unscaled(25, false));
    data.append(FixedTrait::new_unscaled(26, false));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}

fn fp_tensor_3x3x3_neg_helper() -> Tensor<FP16x16> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));
    data.append(FixedTrait::new_unscaled(6, true));
    data.append(FixedTrait::new_unscaled(7, true));
    data.append(FixedTrait::new_unscaled(8, true));
    data.append(FixedTrait::new_unscaled(9, true));
    data.append(FixedTrait::new_unscaled(10, true));
    data.append(FixedTrait::new_unscaled(11, true));
    data.append(FixedTrait::new_unscaled(12, true));
    data.append(FixedTrait::new_unscaled(13, true));
    data.append(FixedTrait::new_unscaled(14, true));
    data.append(FixedTrait::new_unscaled(15, true));
    data.append(FixedTrait::new_unscaled(16, true));
    data.append(FixedTrait::new_unscaled(17, true));
    data.append(FixedTrait::new_unscaled(18, true));
    data.append(FixedTrait::new_unscaled(19, true));
    data.append(FixedTrait::new_unscaled(20, true));
    data.append(FixedTrait::new_unscaled(21, true));
    data.append(FixedTrait::new_unscaled(22, true));
    data.append(FixedTrait::new_unscaled(23, true));
    data.append(FixedTrait::new_unscaled(24, true));
    data.append(FixedTrait::new_unscaled(25, true));
    data.append(FixedTrait::new_unscaled(26, true));

    

    let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    return tensor;
}
