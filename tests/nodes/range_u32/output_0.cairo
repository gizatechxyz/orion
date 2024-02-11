use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(4);
    data.append(7);
    data.append(10);
    data.append(13);
    data.append(16);
    data.append(19);
    data.append(22);
    TensorTrait::new(shape.span(), data.span())
}
