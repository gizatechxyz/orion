use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(15);
    data.append(205);
    data.append(192);
    data.append(220);
    data.append(182);
    data.append(184);
    data.append(11);
    data.append(97);
    TensorTrait::new(shape.span(), data.span())
}