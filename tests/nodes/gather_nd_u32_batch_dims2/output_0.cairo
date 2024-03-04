use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(15);
    data.append(24);
    data.append(40);
    data.append(51);
    data.append(61);
    data.append(75);
    data.append(84);
    data.append(100);
    TensorTrait::new(shape.span(), data.span())
}
