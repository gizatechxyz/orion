use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-58);
    data.append(48);
    data.append(-67);
    data.append(-75);
    data.append(28);
    data.append(-1);
    data.append(86);
    data.append(-38);
    data.append(-47);
    data.append(-30);
    data.append(-27);
    data.append(82);
    data.append(-61);
    data.append(-69);
    data.append(-113);
    data.append(45);
    data.append(108);
    data.append(-68);
    TensorTrait::new(shape.span(), data.span())
}
