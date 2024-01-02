use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(9);
    data.append(-26);
    data.append(49);
    data.append(55);
    data.append(28);
    data.append(31);
    data.append(103);
    data.append(-57);
    data.append(65);
    data.append(-10);
    data.append(99);
    data.append(-6);
    data.append(21);
    data.append(-14);
    data.append(78);
    data.append(31);
    data.append(119);
    data.append(-121);
    data.append(44);
    data.append(125);
    TensorTrait::new(shape.span(), data.span())
}
