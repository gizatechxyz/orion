use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(136);
    data.append(153);
    data.append(113);
    data.append(185);
    data.append(177);
    data.append(209);
    data.append(38);
    data.append(211);

    
    TensorTrait::new(shape.span(), data.span())
}