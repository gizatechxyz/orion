use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(4);
    data.append(3);
    data.append(5);
    TensorTrait::new(shape.span(), data.span())
}
