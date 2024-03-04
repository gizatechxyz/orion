use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-92);
    data.append(-127);
    data.append(-47);
    data.append(14);
    data.append(122);
    data.append(5);
    data.append(-35);
    data.append(41);
    data.append(-6);
    data.append(13);
    data.append(118);
    data.append(-24);
    data.append(-104);
    data.append(-92);
    data.append(49);
    TensorTrait::new(shape.span(), data.span())
}
