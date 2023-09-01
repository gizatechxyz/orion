use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};

use orion::operators::tensor::implementations::tensor_u32_fp16x16::Tensor_u32_fp16x16;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(2);
    data.append(5);
    data.append(2);
    data.append(5);
    data.append(0);
    data.append(2);
    data.append(5);
    data.append(1);
    data.append(3);
    data.append(2);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(4);
    data.append(4);
    data.append(0);
    data.append(2);
    data.append(0);
    data.append(2);
    data.append(5);
    data.append(2);
    data.append(1);
    data.append(2);
    data.append(1);
    data.append(2);

    
    TensorTrait::new(shape.span(), data.span())
}