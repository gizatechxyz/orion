use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(178);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(214);
    data.append(206);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(246);
    data.append(248);
    data.append(125);
    data.append(0);
    data.append(0);
    data.append(77);
    data.append(126);
    data.append(22);
    data.append(57);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
