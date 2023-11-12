use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(66);
    data.append(139);
    data.append(229);
    data.append(194);
    data.append(56);
    data.append(188);
    data.append(51);
    data.append(197);
    data.append(237);
    data.append(240);
    data.append(177);
    data.append(112);
    data.append(114);
    data.append(235);
    data.append(200);
    data.append(197);
    data.append(57);
    data.append(67);
    data.append(41);
    data.append(62);
    TensorTrait::new(shape.span(), data.span())
}
