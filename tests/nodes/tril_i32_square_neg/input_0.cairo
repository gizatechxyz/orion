use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-126);
    data.append(120);
    data.append(-98);
    data.append(-22);
    data.append(-68);
    data.append(-104);
    data.append(-51);
    data.append(13);
    data.append(-3);
    data.append(-60);
    data.append(-23);
    data.append(38);
    data.append(-12);
    data.append(114);
    data.append(-63);
    data.append(64);
    data.append(105);
    data.append(-41);
    TensorTrait::new(shape.span(), data.span())
}
