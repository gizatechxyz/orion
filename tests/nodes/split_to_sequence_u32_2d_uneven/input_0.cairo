use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(181);
    data.append(95);
    data.append(164);
    data.append(86);
    data.append(6);
    data.append(169);
    data.append(184);
    data.append(122);
    data.append(132);
    data.append(59);
    data.append(125);
    data.append(118);
    data.append(247);
    data.append(59);
    data.append(17);
    data.append(130);
    TensorTrait::new(shape.span(), data.span())
}
