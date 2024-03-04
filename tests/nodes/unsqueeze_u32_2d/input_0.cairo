use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorSub};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(227);
    data.append(225);
    data.append(104);
    data.append(14);
    data.append(245);
    data.append(235);
    data.append(124);
    data.append(11);
    TensorTrait::new(shape.span(), data.span())
}
