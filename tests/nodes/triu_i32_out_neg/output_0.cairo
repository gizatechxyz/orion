use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(-59);
    data.append(-115);
    data.append(105);
    data.append(-12);
    data.append(34);
    data.append(53);
    data.append(103);
    data.append(-104);
    data.append(9);
    data.append(-85);
    data.append(82);
    data.append(100);
    data.append(-47);
    data.append(39);
    data.append(92);
    data.append(32);
    data.append(-117);
    data.append(121);
    data.append(-107);
    data.append(-60);
    TensorTrait::new(shape.span(), data.span())
}
