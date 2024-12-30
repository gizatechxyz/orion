use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(2712);
    data.append(2520);
    data.append(1656);
    data.append(20114);
    data.append(18690);
    data.append(12282);
    data.append(21131);
    data.append(19635);
    data.append(12903);
    TensorTrait::new(shape.span(), data.span())
}
