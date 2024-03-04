use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(86);
    data.append(-82);
    data.append(57);
    data.append(-15);
    data.append(100);
    data.append(93);
    data.append(48);
    data.append(-18);
    data.append(-115);
    data.append(32);
    data.append(39);
    data.append(103);
    data.append(-87);
    data.append(62);
    data.append(-27);
    data.append(118);
    data.append(-94);
    data.append(122);
    data.append(0);
    data.append(49);
    TensorTrait::new(shape.span(), data.span())
}
