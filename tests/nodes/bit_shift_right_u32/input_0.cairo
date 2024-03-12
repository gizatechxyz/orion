use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(57);
    data.append(22);
    data.append(6);
    data.append(7);
    data.append(4);
    data.append(23);
    data.append(40);
    data.append(31);
    data.append(24);
    TensorTrait::new(shape.span(), data.span())
}
