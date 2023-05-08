use array::ArrayTrait;
use array::SpanTrait;


use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::tensor::implementations::impl_tensor_i32;
use onnx_cairo::operators::tensor::core::{TensorTrait, Tensor};

// 1D
fn i32_tensor_1x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

// 2D

fn i32_tensor_2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));
    data.append(IntegerTrait::new(4_u32, false));
    data.append(IntegerTrait::new(5_u32, false));
    data.append(IntegerTrait::new(6_u32, false));
    data.append(IntegerTrait::new(7_u32, false));
    data.append(IntegerTrait::new(8_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));
    data.append(IntegerTrait::new(4_u32, false));
    data.append(IntegerTrait::new(5_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_2x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));
    data.append(IntegerTrait::new(4_u32, false));
    data.append(IntegerTrait::new(5_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

// 3D

fn i32_tensor_2x2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));
    data.append(IntegerTrait::new(4_u32, false));
    data.append(IntegerTrait::new(5_u32, false));
    data.append(IntegerTrait::new(6_u32, false));
    data.append(IntegerTrait::new(7_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    data.append(IntegerTrait::new(3_u32, false));
    data.append(IntegerTrait::new(4_u32, false));
    data.append(IntegerTrait::new(5_u32, false));
    data.append(IntegerTrait::new(6_u32, false));
    data.append(IntegerTrait::new(7_u32, false));
    data.append(IntegerTrait::new(8_u32, false));
    data.append(IntegerTrait::new(9_u32, false));
    data.append(IntegerTrait::new(10_u32, false));
    data.append(IntegerTrait::new(11_u32, false));

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

