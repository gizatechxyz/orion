use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(20);
    data.append(20);
    data.append(20);
    data.append(20);
    data.append(20);
    data.append(20);
    data.append(20);
    data.append(20);
    TensorTrait::new(shape.span(), data.span())
}