use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(42);
    data.append(0);
    data.append(0);
    data.append(-67);
    data.append(-5);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(-111);
    data.append(0);
    data.append(0);
    data.append(-3);
    data.append(32);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
