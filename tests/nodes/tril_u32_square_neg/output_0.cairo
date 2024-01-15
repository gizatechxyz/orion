use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(129);
    data.append(0);
    data.append(0);
    data.append(210);
    data.append(54);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(244);
    data.append(0);
    data.append(0);
    data.append(69);
    data.append(65);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
