use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(218);
    data.append(99);
    data.append(52);
    data.append(215);
    data.append(26);
    data.append(136);
    data.append(89);
    data.append(77);
    data.append(135);
    data.append(4);
    data.append(38);
    data.append(244);
    data.append(15);
    data.append(151);
    data.append(237);
    data.append(108);
    data.append(199);
    data.append(118);
    data.append(51);
    data.append(132);
    TensorTrait::new(shape.span(), data.span())
}
