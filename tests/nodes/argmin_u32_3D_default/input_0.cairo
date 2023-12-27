use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(132);
    data.append(217);
    data.append(50);
    data.append(22);
    data.append(79);
    data.append(215);
    data.append(166);
    data.append(125);
    TensorTrait::new(shape.span(), data.span())
}
