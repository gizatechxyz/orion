use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-69);
    data.append(51);
    data.append(-49);
    TensorTrait::new(shape.span(), data.span())
}
