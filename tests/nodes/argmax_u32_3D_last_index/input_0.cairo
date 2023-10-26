use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(39);
    data.append(86);
    data.append(127);
    data.append(185);
    data.append(36);
    data.append(197);
    data.append(160);
    data.append(37);
    TensorTrait::new(shape.span(), data.span())
}
