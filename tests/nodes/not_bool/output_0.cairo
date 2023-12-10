use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::BoolTensor;

fn output_0() -> Tensor<bool> {
    let mut shape = ArrayTrait::new();
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(false);
    TensorTrait::new(shape.span(), data.span())
}
