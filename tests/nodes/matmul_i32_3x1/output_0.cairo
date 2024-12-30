use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(170);
    data.append(1564);
    data.append(255);
    data.append(-360);
    data.append(-3312);
    data.append(-540);
    data.append(-510);
    data.append(-4692);
    data.append(-765);
    TensorTrait::new(shape.span(), data.span())
}
