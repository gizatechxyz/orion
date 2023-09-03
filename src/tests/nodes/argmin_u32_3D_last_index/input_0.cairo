use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(238);
    data.append(240);
    data.append(209);
    data.append(189);
    data.append(242);
    data.append(56);
    data.append(128);
    data.append(113);

    
    TensorTrait::new(shape.span(), data.span())
}