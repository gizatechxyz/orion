use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(20);
    data.append(186);
    data.append(209);
    data.append(1);
    data.append(82);
    data.append(184);
    data.append(237);
    data.append(235);
    data.append(142);
    data.append(9);
    data.append(73);
    data.append(90);
    data.append(71);
    data.append(127);
    data.append(102);
    data.append(28);
    data.append(171);
    data.append(121);
    TensorTrait::new(shape.span(), data.span())
}
