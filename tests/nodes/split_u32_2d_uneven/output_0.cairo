use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Array<Tensor<u32>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(114);
    data.append(94);
    data.append(130);
    data.append(173);
    data.append(181);
    data.append(108);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(214);
    data.append(213);
    data.append(226);
    data.append(140);
    data.append(123);
    data.append(14);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(218);
    data.append(47);
    data.append(181);
    data.append(7);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
