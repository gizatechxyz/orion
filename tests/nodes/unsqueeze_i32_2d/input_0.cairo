use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorSub};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(-17);
    data.append(-37);
    data.append(-104);
    data.append(-107);
    data.append(-120);
    data.append(-2);
    data.append(-34);
    data.append(41);
    TensorTrait::new(shape.span(), data.span())
}
