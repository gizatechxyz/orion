use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(101);
    data.append(-95);
    data.append(9);
    data.append(75);
    data.append(-113);
    data.append(108);
    data.append(25);
    data.append(96);
    data.append(115);
    data.append(103);
    data.append(92);
    data.append(-1);
    data.append(-20);
    data.append(60);
    data.append(54);
    data.append(89);
    data.append(18);
    data.append(1);
    data.append(-95);
    data.append(-86);
    TensorTrait::new(shape.span(), data.span())
}
