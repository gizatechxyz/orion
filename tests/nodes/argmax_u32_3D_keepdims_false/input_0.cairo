use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(37);
    data.append(156);
    data.append(115);
    data.append(212);
    data.append(45);
    data.append(237);
    data.append(24);
    data.append(136);
    TensorTrait::new(shape.span(), data.span())
}
