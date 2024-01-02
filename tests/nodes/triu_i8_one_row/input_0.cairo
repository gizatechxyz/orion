use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(12);
    data.append(-9);
    data.append(55);
    data.append(7);
    data.append(0);
    data.append(-20);
    data.append(28);
    data.append(35);
    data.append(47);
    data.append(-121);
    data.append(109);
    data.append(68);
    data.append(-29);
    data.append(-52);
    data.append(-76);
    TensorTrait::new(shape.span(), data.span())
}
