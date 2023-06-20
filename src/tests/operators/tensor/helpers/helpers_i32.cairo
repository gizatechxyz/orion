use array::ArrayTrait;
use array::SpanTrait;


use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

// 1D
fn i32_tensor_1x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

// 2D

fn i32_tensor_2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

fn i32_tensor_3x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(4, false));
    data.append(IntegerTrait::new(5, false));
    data.append(IntegerTrait::new(6, false));
    data.append(IntegerTrait::new(7, false));
    data.append(IntegerTrait::new(8, false));

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

fn i32_tensor_3x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(4, false));
    data.append(IntegerTrait::new(5, false));

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

fn i32_tensor_3x1_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

fn i32_tensor_2x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(4, false));
    data.append(IntegerTrait::new(5, false));

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

// 3D

fn i32_tensor_2x2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(4, false));
    data.append(IntegerTrait::new(5, false));
    data.append(IntegerTrait::new(6, false));
    data.append(IntegerTrait::new(7, false));

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

fn i32_tensor_3x2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0, false));
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(4, false));
    data.append(IntegerTrait::new(5, false));
    data.append(IntegerTrait::new(6, false));
    data.append(IntegerTrait::new(7, false));
    data.append(IntegerTrait::new(8, false));
    data.append(IntegerTrait::new(9, false));
    data.append(IntegerTrait::new(10, false));
    data.append(IntegerTrait::new(11, false));

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

    return tensor;
}

