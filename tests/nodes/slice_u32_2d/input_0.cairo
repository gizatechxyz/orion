use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorSub};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(169);
    data.append(194);
    data.append(34);
    data.append(124);
    data.append(25);
    data.append(12);
    data.append(156);
    data.append(35);
    TensorTrait::new(shape.span(), data.span())
}
