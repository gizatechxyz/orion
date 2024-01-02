use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-64);
    data.append(-112);
    data.append(-109);
    data.append(67);
    data.append(-66);
    data.append(-126);
    data.append(-97);
    data.append(-60);
    data.append(-34);
    data.append(103);
    data.append(112);
    data.append(-41);
    data.append(-93);
    data.append(-72);
    data.append(-6);
    data.append(42);
    data.append(58);
    data.append(4);
    data.append(16);
    data.append(-111);
    TensorTrait::new(shape.span(), data.span())
}
