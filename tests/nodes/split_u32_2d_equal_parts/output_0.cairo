use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Array<Tensor<u32>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(189);
    data.append(74);
    data.append(230);
    data.append(11);
    data.append(159);
    data.append(108);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(245);
    data.append(231);
    data.append(162);
    data.append(92);
    data.append(6);
    data.append(61);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
