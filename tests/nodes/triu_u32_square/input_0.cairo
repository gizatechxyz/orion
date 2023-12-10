use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(221);
    data.append(99);
    data.append(210);
    data.append(132);
    data.append(78);
    data.append(194);
    data.append(197);
    data.append(177);
    data.append(143);
    data.append(17);
    data.append(167);
    data.append(35);
    data.append(146);
    data.append(51);
    data.append(144);
    data.append(72);
    data.append(42);
    data.append(127);
    TensorTrait::new(shape.span(), data.span())
}
