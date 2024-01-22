use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(3);
    data.append(9);
    data.append(15);
    data.append(19);
    data.append(25);
    data.append(31);
    data.append(36);
    data.append(45);
    data.append(50);
    TensorTrait::new(shape.span(), data.span())
}
