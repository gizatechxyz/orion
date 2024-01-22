use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(6);
    data.append(21);
    data.append(29);
    data.append(37);
    data.append(64);
    data.append(54);
    data.append(96);
    data.append(84);
    TensorTrait::new(shape.span(), data.span())
}
