use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(156);
    data.append(143);
    data.append(86);
    data.append(94);
    data.append(20);
    data.append(48);
    data.append(49);
    data.append(138);
    data.append(35);
    data.append(5);
    data.append(163);
    data.append(87);
    data.append(6);
    data.append(105);
    data.append(130);
    data.append(113);
    data.append(72);
    data.append(252);
    data.append(202);
    data.append(184);
    TensorTrait::new(shape.span(), data.span())
}
