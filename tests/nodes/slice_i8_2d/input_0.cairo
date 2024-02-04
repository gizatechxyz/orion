use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(90);
    data.append(-56);
    data.append(-97);
    data.append(83);
    data.append(-82);
    data.append(120);
    data.append(-97);
    data.append(55);
    TensorTrait::new(shape.span(), data.span())
}
