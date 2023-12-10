use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(109);
    data.append(158);
    data.append(245);
    data.append(103);
    data.append(198);
    data.append(0);
    data.append(205);
    data.append(10);
    data.append(11);
    data.append(136);
    data.append(0);
    data.append(0);
    data.append(38);
    data.append(75);
    data.append(181);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(247);
    data.append(124);
    TensorTrait::new(shape.span(), data.span())
}
