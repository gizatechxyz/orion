use array::ArrayTrait;
use array::SpanTrait;


use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};

// 1D
fn i32_tensor_1x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_1x3_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });
    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

// 2D

fn i32_tensor_2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: false });
    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_2x2_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 3, sign: true });
    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 6, sign: false });
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 8, sign: false });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x3_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0_u32, sign: false });
    data.append(i32 { mag: 1_u32, sign: true });
    data.append(i32 { mag: 2_u32, sign: true });
    data.append(i32 { mag: 3_u32, sign: true });
    data.append(i32 { mag: 4_u32, sign: true });
    data.append(i32 { mag: 5_u32, sign: true });
    data.append(i32 { mag: 6_u32, sign: true });
    data.append(i32 { mag: 7_u32, sign: true });
    data.append(i32 { mag: 8_u32, sign: true });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 5, sign: false });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x2_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 5, sign: true });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x1_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x1_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_2x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 5, sign: false });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_2x3_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 5, sign: true });

    

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
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 6, sign: false });
    data.append(i32 { mag: 7, sign: false });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_2x2x2_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0_u32, sign: false });
    data.append(i32 { mag: 1_u32, sign: true });
    data.append(i32 { mag: 2_u32, sign: true });
    data.append(i32 { mag: 3_u32, sign: true });
    data.append(i32 { mag: 4_u32, sign: true });
    data.append(i32 { mag: 5_u32, sign: true });
    data.append(i32 { mag: 6_u32, sign: true });
    data.append(i32 { mag: 7_u32, sign: true });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x2x2_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 6, sign: false });
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 8, sign: false });
    data.append(i32 { mag: 9, sign: false });
    data.append(i32 { mag: 10, sign: false });
    data.append(i32 { mag: 11, sign: false });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x2x2_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 5, sign: true });
    data.append(i32 { mag: 6, sign: true });
    data.append(i32 { mag: 7, sign: true });
    data.append(i32 { mag: 8, sign: true });
    data.append(i32 { mag: 9, sign: true });
    data.append(i32 { mag: 10, sign: true });
    data.append(i32 { mag: 11, sign: true });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x3x3_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 6, sign: false });
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 8, sign: false });
    data.append(i32 { mag: 9, sign: false });
    data.append(i32 { mag: 10, sign: false });
    data.append(i32 { mag: 11, sign: false });
    data.append(i32 { mag: 12, sign: false });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 14, sign: false });
    data.append(i32 { mag: 15, sign: false });
    data.append(i32 { mag: 16, sign: false });
    data.append(i32 { mag: 17, sign: false });
    data.append(i32 { mag: 18, sign: false });
    data.append(i32 { mag: 19, sign: false });
    data.append(i32 { mag: 20, sign: false });
    data.append(i32 { mag: 21, sign: false });
    data.append(i32 { mag: 22, sign: false });
    data.append(i32 { mag: 23, sign: false });
    data.append(i32 { mag: 24, sign: false });
    data.append(i32 { mag: 25, sign: false });
    data.append(i32 { mag: 26, sign: false });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}

fn i32_tensor_3x3x3_neg_helper() -> Tensor<i32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 5, sign: true });
    data.append(i32 { mag: 6, sign: true });
    data.append(i32 { mag: 7, sign: true });
    data.append(i32 { mag: 8, sign: true });
    data.append(i32 { mag: 9, sign: true });
    data.append(i32 { mag: 10, sign: true });
    data.append(i32 { mag: 11, sign: true });
    data.append(i32 { mag: 12, sign: true });
    data.append(i32 { mag: 13, sign: true });
    data.append(i32 { mag: 14, sign: true });
    data.append(i32 { mag: 15, sign: true });
    data.append(i32 { mag: 16, sign: true });
    data.append(i32 { mag: 17, sign: true });
    data.append(i32 { mag: 18, sign: true });
    data.append(i32 { mag: 19, sign: true });
    data.append(i32 { mag: 20, sign: true });
    data.append(i32 { mag: 21, sign: true });
    data.append(i32 { mag: 22, sign: true });
    data.append(i32 { mag: 23, sign: true });
    data.append(i32 { mag: 24, sign: true });
    data.append(i32 { mag: 25, sign: true });
    data.append(i32 { mag: 26, sign: true });

    

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span());

    return tensor;
}
