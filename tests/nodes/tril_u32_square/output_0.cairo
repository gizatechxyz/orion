use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(14);
    data.append(0);
    data.append(0);
    data.append(34);
    data.append(216);
    data.append(0);
    data.append(132);
    data.append(143);
    data.append(161);
    data.append(9);
    data.append(0);
    data.append(0);
    data.append(250);
    data.append(157);
    data.append(0);
    data.append(32);
    data.append(147);
    data.append(90);
    TensorTrait::new(shape.span(), data.span())
}
