use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(117);
    data.append(-107);
    data.append(-33);
    data.append(-116);
    data.append(-69);
    data.append(124);
    data.append(71);
    data.append(106);
    data.append(-37);
    data.append(10);
    data.append(-46);
    data.append(44);
    data.append(-120);
    data.append(-92);
    data.append(-121);
    data.append(60);
    data.append(53);
    data.append(-15);
    data.append(-101);
    data.append(121);
    TensorTrait::new(shape.span(), data.span())
}
