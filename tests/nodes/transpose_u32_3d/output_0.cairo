use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorSub};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(93);
    data.append(212);
    data.append(120);
    data.append(82);
    data.append(177);
    data.append(91);
    data.append(21);
    data.append(240);
    TensorTrait::new(shape.span(), data.span())
}
