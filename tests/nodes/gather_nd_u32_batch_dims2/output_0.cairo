use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(16);
    data.append(27);
    data.append(39);
    data.append(51);
    data.append(64);
    data.append(73);
    data.append(84);
    data.append(97);
    TensorTrait::new(shape.span(), data.span())
}
