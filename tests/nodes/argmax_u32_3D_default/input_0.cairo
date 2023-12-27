use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(22);
    data.append(254);
    data.append(48);
    data.append(151);
    data.append(21);
    data.append(13);
    data.append(254);
    data.append(100);
    TensorTrait::new(shape.span(), data.span())
}
