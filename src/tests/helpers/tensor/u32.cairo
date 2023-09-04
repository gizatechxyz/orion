use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};

// 1D
fn u32_tensor_1x3_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

// 2D

fn u32_tensor_2x2_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_3x3_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);

    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_3x2_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);

    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_3x1_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);

    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_2x3_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);

    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

// 3D

fn u32_tensor_2x2x2_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);

    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_3x2x2_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    data.append(11);

    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

fn u32_tensor_3x3x3_helper() -> Tensor<u32> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    data.append(11);
    data.append(12);
    data.append(13);
    data.append(14);
    data.append(15);
    data.append(16);
    data.append(17);
    data.append(18);
    data.append(19);
    data.append(20);
    data.append(21);
    data.append(22);
    data.append(23);
    data.append(24);
    data.append(25);
    data.append(26);

    

    let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());

    return tensor;
}

