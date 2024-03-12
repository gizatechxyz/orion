use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(336);
    data.append(7);
    data.append(18);
    data.append(344);
    data.append(98);
    data.append(98);
    data.append(8);
    data.append(56);
    data.append(24);
    TensorTrait::new(shape.span(), data.span())
}
