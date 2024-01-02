use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(0);
    data.append(4294967295);
    data.append(8);
    data.append(1);
    data.append(4294967295);
    TensorTrait::new(shape.span(), data.span())
}
