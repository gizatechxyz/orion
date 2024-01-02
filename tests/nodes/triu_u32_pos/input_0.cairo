use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(244);
    data.append(163);
    data.append(108);
    data.append(70);
    data.append(20);
    data.append(91);
    data.append(177);
    data.append(213);
    data.append(72);
    data.append(36);
    data.append(146);
    data.append(21);
    data.append(61);
    data.append(167);
    data.append(124);
    data.append(6);
    data.append(73);
    data.append(25);
    data.append(72);
    data.append(210);
    TensorTrait::new(shape.span(), data.span())
}
