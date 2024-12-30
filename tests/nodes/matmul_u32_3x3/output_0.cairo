use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(29378);
    data.append(24503);
    data.append(25663);
    data.append(59522);
    data.append(38139);
    data.append(48825);
    data.append(40016);
    data.append(68147);
    data.append(34795);
    TensorTrait::new(shape.span(), data.span())
}
