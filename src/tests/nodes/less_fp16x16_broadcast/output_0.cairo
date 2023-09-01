use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};

use orion::operators::tensor::implementations::tensor_u32_fp16x16::Tensor_u32_fp16x16;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(1);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(1);
    data.append(0);
    data.append(0);

    
    TensorTrait::new(shape.span(), data.span())
}