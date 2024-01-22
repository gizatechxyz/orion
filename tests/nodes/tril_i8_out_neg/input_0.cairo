use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(67);
    data.append(-102);
    data.append(45);
    data.append(77);
    data.append(55);
    data.append(-19);
    data.append(-56);
    data.append(-22);
    data.append(-32);
    data.append(102);
    data.append(23);
    data.append(72);
    data.append(-34);
    data.append(62);
    data.append(67);
    data.append(45);
    data.append(-49);
    data.append(108);
    data.append(-83);
    data.append(22);
    TensorTrait::new(shape.span(), data.span())
}
