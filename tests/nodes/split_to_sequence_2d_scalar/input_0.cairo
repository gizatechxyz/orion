use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(230);
    data.append(131);
    data.append(175);
    data.append(106);
    data.append(106);
    data.append(44);
    data.append(254);
    data.append(157);
    data.append(131);
    data.append(251);
    data.append(38);
    data.append(14);
    data.append(0);
    data.append(116);
    data.append(225);
    data.append(107);
    TensorTrait::new(shape.span(), data.span())
}
