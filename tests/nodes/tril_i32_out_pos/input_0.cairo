use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(5);
    data.append(11);
    data.append(21);
    data.append(-27);
    data.append(98);
    data.append(108);
    data.append(65);
    data.append(-4);
    data.append(7);
    data.append(-33);
    data.append(109);
    data.append(-55);
    data.append(0);
    data.append(-71);
    data.append(-108);
    data.append(-5);
    data.append(-40);
    data.append(75);
    data.append(69);
    data.append(66);
    TensorTrait::new(shape.span(), data.span())
}
