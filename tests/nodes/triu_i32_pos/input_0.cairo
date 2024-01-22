use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-19);
    data.append(-28);
    data.append(44);
    data.append(-26);
    data.append(45);
    data.append(83);
    data.append(78);
    data.append(110);
    data.append(89);
    data.append(78);
    data.append(30);
    data.append(-22);
    data.append(73);
    data.append(-114);
    data.append(114);
    data.append(-25);
    data.append(-6);
    data.append(3);
    data.append(9);
    data.append(79);
    TensorTrait::new(shape.span(), data.span())
}
