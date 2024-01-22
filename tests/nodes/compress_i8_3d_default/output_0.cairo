use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(9);
    data.append(10);
    data.append(11);
    data.append(12);
    data.append(13);
    data.append(14);
    data.append(15);
    data.append(16);
    data.append(17);
    data.append(18);
    data.append(19);
    data.append(20);
    data.append(21);
    data.append(22);
    data.append(23);
    data.append(24);
    data.append(25);
    data.append(26);
    TensorTrait::new(shape.span(), data.span())
}
