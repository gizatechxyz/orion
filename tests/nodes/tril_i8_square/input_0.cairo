use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(120);
    data.append(54);
    data.append(12);
    data.append(62);
    data.append(46);
    data.append(53);
    data.append(-119);
    data.append(-9);
    data.append(58);
    data.append(-69);
    data.append(97);
    data.append(99);
    data.append(-23);
    data.append(-19);
    data.append(5);
    data.append(-58);
    data.append(72);
    data.append(-11);
    TensorTrait::new(shape.span(), data.span())
}
