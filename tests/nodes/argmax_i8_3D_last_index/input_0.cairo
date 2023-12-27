use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-7);
    data.append(-94);
    data.append(-64);
    data.append(-19);
    data.append(59);
    data.append(-40);
    data.append(99);
    data.append(38);
    TensorTrait::new(shape.span(), data.span())
}
