use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(21);
    data.append(18);
    data.append(15);
    data.append(12);
    data.append(9);
    data.append(6);
    data.append(3);
    TensorTrait::new(shape.span(), data.span())
}
