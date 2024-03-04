use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(72);
    data.append(209);
    data.append(27);
    data.append(147);
    data.append(22);
    data.append(98);
    data.append(135);
    TensorTrait::new(shape.span(), data.span())
}
