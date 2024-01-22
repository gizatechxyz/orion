use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-71);
    data.append(99);
    data.append(106);
    data.append(-4);
    data.append(-110);
    data.append(56);
    data.append(-77);
    data.append(-6);
    TensorTrait::new(shape.span(), data.span())
}
