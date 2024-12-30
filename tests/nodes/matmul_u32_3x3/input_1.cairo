use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(69);
    data.append(239);
    data.append(34);
    data.append(181);
    data.append(78);
    data.append(99);
    data.append(160);
    data.append(74);
    data.append(236);
    TensorTrait::new(shape.span(), data.span())
}
