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
    data.append(156);
    data.append(136);
    data.append(146);
    data.append(150);
    data.append(69);
    data.append(43);
    data.append(25);

    
    TensorTrait::new(shape.span(), data.span())
}