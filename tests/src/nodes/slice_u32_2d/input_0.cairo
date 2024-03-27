use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(239);
    data.append(177);
    data.append(83);
    data.append(88);
    data.append(34);
    data.append(131);
    data.append(2);
    data.append(11);
    TensorTrait::new(shape.span(), data.span())
}
