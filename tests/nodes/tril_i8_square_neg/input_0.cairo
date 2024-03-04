use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(111);
    data.append(83);
    data.append(82);
    data.append(42);
    data.append(0);
    data.append(3);
    data.append(-67);
    data.append(-5);
    data.append(-78);
    data.append(-17);
    data.append(-46);
    data.append(-65);
    data.append(-111);
    data.append(71);
    data.append(13);
    data.append(-3);
    data.append(32);
    data.append(-34);
    TensorTrait::new(shape.span(), data.span())
}
