use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(230);
    data.append(202);
    data.append(125);
    data.append(133);
    data.append(198);
    data.append(253);
    data.append(218);
    data.append(151);
    data.append(55);
    data.append(71);
    data.append(87);
    data.append(62);
    data.append(126);
    data.append(86);
    data.append(87);
    data.append(173);
    data.append(170);
    data.append(166);
    TensorTrait::new(shape.span(), data.span())
}
