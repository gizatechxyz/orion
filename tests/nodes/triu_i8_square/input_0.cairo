use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(34);
    data.append(32);
    data.append(-105);
    data.append(-102);
    data.append(36);
    data.append(-12);
    data.append(-112);
    data.append(76);
    data.append(-119);
    data.append(71);
    data.append(48);
    data.append(77);
    data.append(-117);
    data.append(-19);
    data.append(15);
    data.append(100);
    data.append(55);
    data.append(30);
    TensorTrait::new(shape.span(), data.span())
}
