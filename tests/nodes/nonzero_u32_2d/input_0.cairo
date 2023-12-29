use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorSub};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(75);
    data.append(210);
    data.append(169);
    data.append(35);
    data.append(213);
    data.append(12);
    data.append(32);
    data.append(9);
    TensorTrait::new(shape.span(), data.span())
}
