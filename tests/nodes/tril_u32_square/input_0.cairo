use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(145);
    data.append(153);
    data.append(135);
    data.append(19);
    data.append(97);
    data.append(193);
    data.append(184);
    data.append(77);
    data.append(193);
    data.append(158);
    data.append(160);
    data.append(144);
    data.append(199);
    data.append(30);
    data.append(71);
    data.append(99);
    data.append(244);
    data.append(112);
    TensorTrait::new(shape.span(), data.span())
}
