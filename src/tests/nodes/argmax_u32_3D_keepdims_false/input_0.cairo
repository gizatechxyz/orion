use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(126);
    data.append(36);
    data.append(190);
    data.append(190);
    data.append(114);
    data.append(158);
    data.append(103);
    data.append(89);

    
    TensorTrait::new(shape.span(), data.span())
}