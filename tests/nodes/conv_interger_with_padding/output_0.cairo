use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(4);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(3);
    data.append(5);
    data.append(3);
    data.append(5);
    data.append(12);
    data.append(16);
    data.append(9);
    data.append(11);
    data.append(24);
    data.append(28);
    data.append(15);
    data.append(7);
    data.append(15);
    data.append(17);
    data.append(9);
    TensorTrait::new(shape.span(), data.span())
}
