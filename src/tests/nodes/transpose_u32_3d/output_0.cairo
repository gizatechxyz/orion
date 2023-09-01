use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};

use orion::operators::tensor::implementations::tensor_u32_fp16x16::Tensor_u32_fp16x16;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(180);
    data.append(61);
    data.append(20);
    data.append(240);
    data.append(198);
    data.append(165);
    data.append(208);
    data.append(75);

    
    TensorTrait::new(shape.span(), data.span())
}