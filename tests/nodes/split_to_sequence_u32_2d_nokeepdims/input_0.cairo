use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(88);
    data.append(30);
    data.append(151);
    data.append(224);
    data.append(178);
    data.append(125);
    data.append(93);
    data.append(64);
    data.append(27);
    data.append(216);
    data.append(220);
    data.append(33);
    data.append(155);
    data.append(107);
    data.append(190);
    data.append(142);
    data.append(56);
    data.append(224);
    TensorTrait::new(shape.span(), data.span())
}
