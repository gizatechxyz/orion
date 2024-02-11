use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(67);
    data.append(177);
    data.append(5);
    data.append(93);
    data.append(183);
    data.append(173);
    data.append(207);
    data.append(194);
    TensorTrait::new(shape.span(), data.span())
}
