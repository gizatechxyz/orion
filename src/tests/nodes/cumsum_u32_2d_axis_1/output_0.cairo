use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(3);
    data.append(6);
    data.append(4);
    data.append(9);
    data.append(15);
    TensorTrait::new(shape.span(), data.span())
}