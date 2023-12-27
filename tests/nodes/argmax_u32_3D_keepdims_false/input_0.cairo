use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(183);
    data.append(138);
    data.append(72);
    data.append(104);
    data.append(239);
    data.append(185);
    data.append(16);
    data.append(188);
    TensorTrait::new(shape.span(), data.span())
}
