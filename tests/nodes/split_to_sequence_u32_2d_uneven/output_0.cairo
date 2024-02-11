use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Array<Tensor<u32>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(181);
    data.append(95);
    data.append(164);
    data.append(132);
    data.append(59);
    data.append(125);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(86);
    data.append(6);
    data.append(169);
    data.append(118);
    data.append(247);
    data.append(59);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(184);
    data.append(122);
    data.append(17);
    data.append(130);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
