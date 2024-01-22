use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(178);
    data.append(133);
    data.append(129);
    data.append(130);
    data.append(51);
    data.append(214);
    data.append(206);
    data.append(103);
    data.append(71);
    data.append(19);
    data.append(246);
    data.append(248);
    data.append(125);
    data.append(193);
    data.append(81);
    data.append(77);
    data.append(126);
    data.append(22);
    data.append(57);
    data.append(247);
    TensorTrait::new(shape.span(), data.span())
}
