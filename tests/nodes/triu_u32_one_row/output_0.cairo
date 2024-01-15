use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(198);
    data.append(203);
    data.append(245);
    data.append(39);
    data.append(194);
    data.append(11);
    data.append(155);
    data.append(73);
    data.append(219);
    data.append(69);
    data.append(96);
    data.append(34);
    data.append(45);
    data.append(175);
    data.append(211);
    TensorTrait::new(shape.span(), data.span())
}
