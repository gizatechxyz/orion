use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(9);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(31);
    data.append(103);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(99);
    data.append(-6);
    data.append(21);
    data.append(0);
    data.append(0);
    data.append(31);
    data.append(119);
    data.append(-121);
    data.append(44);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
