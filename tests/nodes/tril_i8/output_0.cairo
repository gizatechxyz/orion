use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-83);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(-100);
    data.append(67);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(-38);
    data.append(-71);
    data.append(-49);
    data.append(0);
    data.append(0);
    data.append(99);
    data.append(-92);
    data.append(105);
    data.append(92);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
