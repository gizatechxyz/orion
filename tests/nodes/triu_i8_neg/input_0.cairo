use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(41);
    data.append(75);
    data.append(6);
    data.append(119);
    data.append(-65);
    data.append(76);
    data.append(-117);
    data.append(91);
    data.append(-83);
    data.append(72);
    data.append(25);
    data.append(-28);
    data.append(-102);
    data.append(77);
    data.append(126);
    data.append(88);
    data.append(-38);
    data.append(115);
    data.append(44);
    data.append(-23);
    TensorTrait::new(shape.span(), data.span())
}
