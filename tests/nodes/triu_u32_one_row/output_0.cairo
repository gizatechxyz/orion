use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(139);
    data.append(223);
    data.append(129);
    data.append(83);
    data.append(41);
    data.append(88);
    data.append(145);
    data.append(7);
    data.append(203);
    data.append(124);
    data.append(5);
    data.append(112);
    data.append(61);
    data.append(77);
    data.append(207);
    TensorTrait::new(shape.span(), data.span())
}
