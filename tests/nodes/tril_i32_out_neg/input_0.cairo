use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(110);
    data.append(-94);
    data.append(103);
    data.append(-52);
    data.append(42);
    data.append(-26);
    data.append(-31);
    data.append(-31);
    data.append(-81);
    data.append(117);
    data.append(-17);
    data.append(70);
    data.append(-96);
    data.append(123);
    data.append(97);
    data.append(-3);
    data.append(76);
    data.append(-75);
    data.append(-26);
    data.append(26);
    TensorTrait::new(shape.span(), data.span())
}
