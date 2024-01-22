use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-66);
    data.append(-82);
    data.append(10);
    data.append(-17);
    data.append(109);
    data.append(-65);
    data.append(0);
    data.append(34);
    data.append(-96);
    data.append(90);
    data.append(-123);
    data.append(45);
    data.append(47);
    data.append(-107);
    data.append(-116);
    data.append(0);
    data.append(-13);
    data.append(94);
    TensorTrait::new(shape.span(), data.span())
}
