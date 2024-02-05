use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Array<Tensor<u32>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(245);
    data.append(158);
    data.append(64);
    data.append(77);
    data.append(139);
    data.append(112);
    data.append(17);
    data.append(94);
    data.append(214);
    data.append(186);
    data.append(66);
    data.append(249);
    data.append(171);
    data.append(110);
    data.append(38);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(20);
    data.append(138);
    data.append(118);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
