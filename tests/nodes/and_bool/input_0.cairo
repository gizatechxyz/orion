use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::BoolTensor;

fn input_0() -> Tensor<bool> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(true);
    data.append(false);
    data.append(false);
    data.append(false);
    data.append(true);
    data.append(false);
    data.append(true);
    data.append(true);
    data.append(true);
    data.append(false);
    data.append(false);
    data.append(false);
    TensorTrait::new(shape.span(), data.span())
}
