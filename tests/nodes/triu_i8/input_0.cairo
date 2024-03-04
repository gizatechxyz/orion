use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(10);
    data.append(78);
    data.append(-79);
    data.append(-74);
    data.append(-116);
    data.append(48);
    data.append(77);
    data.append(85);
    data.append(96);
    data.append(-110);
    data.append(-55);
    data.append(-37);
    data.append(50);
    data.append(-38);
    data.append(117);
    data.append(-34);
    data.append(-8);
    data.append(93);
    data.append(-105);
    data.append(119);
    TensorTrait::new(shape.span(), data.span())
}
