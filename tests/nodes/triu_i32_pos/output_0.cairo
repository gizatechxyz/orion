use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(0);
    data.append(0);
    data.append(44);
    data.append(-26);
    data.append(45);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(89);
    data.append(78);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(114);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(0);
    TensorTrait::new(shape.span(), data.span())
}
