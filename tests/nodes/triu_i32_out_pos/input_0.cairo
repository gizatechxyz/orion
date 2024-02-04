use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-70);
    data.append(-114);
    data.append(-2);
    data.append(-56);
    data.append(102);
    data.append(22);
    data.append(23);
    data.append(-101);
    data.append(119);
    data.append(-122);
    data.append(-82);
    data.append(-102);
    data.append(56);
    data.append(103);
    data.append(51);
    data.append(-78);
    data.append(-120);
    data.append(-111);
    data.append(-12);
    data.append(80);
    TensorTrait::new(shape.span(), data.span())
}
