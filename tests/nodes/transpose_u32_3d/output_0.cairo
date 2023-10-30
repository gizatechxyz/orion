use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(6);
    data.append(104);
    data.append(44);
    data.append(24);
    data.append(94);
    data.append(21);
    data.append(97);
    data.append(83);
    TensorTrait::new(shape.span(), data.span())
}
