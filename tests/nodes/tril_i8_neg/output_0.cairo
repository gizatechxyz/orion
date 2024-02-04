use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(30);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(45);
    data.append(76);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(114);
    data.append(11);
    data.append(-92);
    data.append(0);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
