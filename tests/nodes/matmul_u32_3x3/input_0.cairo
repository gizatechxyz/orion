use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(55);
    data.append(83);
    data.append(66);
    data.append(57);
    data.append(209);
    data.append(111);
    data.append(241);
    data.append(47);
    data.append(93);
    TensorTrait::new(shape.span(), data.span())
}
