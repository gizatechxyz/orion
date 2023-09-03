use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(205);
    data.append(21);
    data.append(93);
    data.append(199);
    data.append(44);
    data.append(76);
    data.append(60);
    data.append(73);

    
    TensorTrait::new(shape.span(), data.span())
}