use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(66);
    data.append(-56);
    data.append(49);
    data.append(-2);
    data.append(-93);
    data.append(-55);
    data.append(115);
    data.append(28);
    TensorTrait::new(shape.span(), data.span())
}
