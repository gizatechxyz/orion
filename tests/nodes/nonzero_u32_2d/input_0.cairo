use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(114);
    data.append(69);
    data.append(131);
    data.append(47);
    data.append(136);
    data.append(205);
    data.append(35);
    data.append(33);
    TensorTrait::new(shape.span(), data.span())
}
