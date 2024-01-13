use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::{NumberTrait};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(-1);
    data.append(0);
    data.append(NumberTrait::INF());
    data.append(8);
    data.append(NumberTrait::INF() * -1);
    data.append(NumberTrait::INF());
    TensorTrait::new(shape.span(), data.span())
}
