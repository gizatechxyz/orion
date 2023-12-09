use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(113);
    data.append(87);
    data.append(46);
    data.append(233);
    data.append(48);
    data.append(184);
    data.append(132);
    data.append(29);
    data.append(137);
    data.append(2);
    data.append(0);
    data.append(152);
    data.append(55);
    data.append(58);
    data.append(190);
    data.append(0);
    data.append(0);
    data.append(204);
    data.append(173);
    data.append(150);
    TensorTrait::new(shape.span(), data.span())
}
