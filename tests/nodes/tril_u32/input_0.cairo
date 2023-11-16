use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(182);
    data.append(80);
    data.append(147);
    data.append(74);
    data.append(49);
    data.append(129);
    data.append(84);
    data.append(164);
    data.append(205);
    data.append(48);
    data.append(150);
    data.append(158);
    data.append(16);
    data.append(127);
    data.append(250);
    data.append(55);
    data.append(177);
    data.append(114);
    data.append(231);
    data.append(47);
    TensorTrait::new(shape.span(), data.span())
}
