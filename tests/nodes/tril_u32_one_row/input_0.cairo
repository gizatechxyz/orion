use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(250);
    data.append(39);
    data.append(83);
    data.append(231);
    data.append(236);
    data.append(69);
    data.append(147);
    data.append(61);
    data.append(66);
    data.append(230);
    data.append(145);
    data.append(231);
    data.append(205);
    data.append(152);
    data.append(157);
    TensorTrait::new(shape.span(), data.span())
}
