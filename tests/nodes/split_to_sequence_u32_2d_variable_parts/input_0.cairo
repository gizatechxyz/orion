use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(24);
    data.append(113);
    data.append(214);
    data.append(210);
    data.append(195);
    data.append(92);
    data.append(187);
    data.append(1);
    data.append(10);
    data.append(135);
    data.append(216);
    data.append(113);
    TensorTrait::new(shape.span(), data.span())
}
