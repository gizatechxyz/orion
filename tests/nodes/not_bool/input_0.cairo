use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::BoolTensor;

fn input_0() -> Tensor<bool> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(true);
    TensorTrait::new(shape.span(), data.span())
}
