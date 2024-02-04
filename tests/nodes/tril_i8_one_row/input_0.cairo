use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-53);
    data.append(66);
    data.append(50);
    data.append(-28);
    data.append(-20);
    data.append(69);
    data.append(117);
    data.append(5);
    data.append(-51);
    data.append(24);
    data.append(-23);
    data.append(21);
    data.append(60);
    data.append(6);
    data.append(13);
    TensorTrait::new(shape.span(), data.span())
}
