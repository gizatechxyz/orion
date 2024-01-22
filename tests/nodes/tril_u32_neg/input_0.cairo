use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(91);
    data.append(44);
    data.append(124);
    data.append(155);
    data.append(1);
    data.append(207);
    data.append(226);
    data.append(206);
    data.append(19);
    data.append(101);
    data.append(6);
    data.append(88);
    data.append(35);
    data.append(127);
    data.append(28);
    data.append(163);
    data.append(160);
    data.append(133);
    data.append(184);
    data.append(189);
    TensorTrait::new(shape.span(), data.span())
}
