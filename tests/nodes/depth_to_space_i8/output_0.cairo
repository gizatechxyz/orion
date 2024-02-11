use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(4);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(-2);
    data.append(0);
    data.append(-3);
    data.append(-1);
    data.append(2);
    data.append(2);
    data.append(-3);
    data.append(0);
    data.append(0);
    data.append(2);
    data.append(-1);
    data.append(1);
    data.append(-2);
    data.append(1);
    data.append(0);
    data.append(1);
    TensorTrait::new(shape.span(), data.span())
}
