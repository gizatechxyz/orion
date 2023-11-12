use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(5);
    data.append(25);
    data.append(61);
    data.append(113);
    data.append(181);
    data.append(265);
    TensorTrait::new(shape.span(), data.span())
}
