use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(9);
    data.append(169);
    data.append(140);
    data.append(99);
    data.append(130);
    data.append(132);
    data.append(79);
    data.append(57);
    TensorTrait::new(shape.span(), data.span())
}
