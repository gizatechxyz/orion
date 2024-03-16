use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(153);
    data.append(28);
    data.append(204);
    data.append(0);
    data.append(69);
    data.append(220);
    data.append(126);
    data.append(80);
    TensorTrait::new(shape.span(), data.span())
}
