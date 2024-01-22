use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(245);
    data.append(184);
    data.append(9);
    data.append(245);
    data.append(161);
    data.append(0);
    data.append(3);
    data.append(17);
    data.append(116);
    data.append(72);
    data.append(0);
    data.append(0);
    data.append(130);
    data.append(230);
    data.append(67);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(243);
    data.append(5);
    TensorTrait::new(shape.span(), data.span())
}
