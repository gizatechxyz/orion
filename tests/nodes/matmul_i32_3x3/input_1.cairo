use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(31);
    data.append(90);
    data.append(114);
    data.append(33);
    data.append(91);
    data.append(56);
    data.append(-73);
    data.append(-58);
    data.append(110);
    TensorTrait::new(shape.span(), data.span())
}
