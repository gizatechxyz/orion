use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(130);
    data.append(164);
    data.append(178);
    data.append(140);
    data.append(56);
    data.append(214);
    data.append(0);
    data.append(173);
    data.append(10);
    data.append(229);
    data.append(182);
    data.append(12);
    data.append(167);
    data.append(150);
    data.append(215);
    data.append(0);
    data.append(37);
    data.append(17);
    TensorTrait::new(shape.span(), data.span())
}
