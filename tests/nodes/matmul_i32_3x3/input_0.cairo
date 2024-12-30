use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(115);
    data.append(103);
    data.append(-72);
    data.append(18);
    data.append(59);
    data.append(115);
    data.append(19);
    data.append(-71);
    data.append(-92);
    TensorTrait::new(shape.span(), data.span())
}
