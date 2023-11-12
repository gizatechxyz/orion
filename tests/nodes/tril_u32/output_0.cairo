use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(182);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(129);
    data.append(84);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(150);
    data.append(158);
    data.append(16);
    data.append(0);
    data.append(0);
    data.append(55);
    data.append(177);
    data.append(114);
    data.append(231);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
