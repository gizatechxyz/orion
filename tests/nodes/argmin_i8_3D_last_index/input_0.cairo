use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-73);
    data.append(-96);
    data.append(-90);
    data.append(9);
    data.append(-99);
    data.append(-18);
    data.append(-49);
    data.append(-30);
    TensorTrait::new(shape.span(), data.span())
}
